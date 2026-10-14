# Project uniurdf
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Tests for the GUI server.

The viewer can write files, so the sandbox tests here are the important ones.
The rest pin the payload contract the browser depends on -- and that the
server's FK is the same FK that produces the export, which is the whole reason
kinematics stays on this side of the wire.
"""

import json
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from uniurdf.diagnostics import inspect_urdf
from uniurdf.kinematics.chain import build_chain
from uniurdf.project import Project, StagedRobot
from uniurdf.robot import load_robot
from uniurdf.spec import AlignmentSpec, parse_spec


def _origin_matrix(rpy, xyz) -> np.ndarray:
    """A URDF ``<origin>`` as a 4x4, using the package's own convention."""
    from uniurdf.transform._linalg import OriginTransform

    return OriginTransform.from_rpy_xyz(rpy, xyz).as_matrix()


def _visual_origins(urdf_bytes: bytes) -> dict[str, dict[str, list[float]]]:
    """First visual-mesh origin per link, read straight from the XML."""
    import xml.etree.ElementTree as ET

    def triple(text: str | None) -> list[float]:
        return [float(v) for v in text.split()] if text else [0.0, 0.0, 0.0]

    out: dict[str, dict[str, list[float]]] = {}
    for link in ET.fromstring(urdf_bytes).findall("link"):
        name = link.get("name")
        for visual in link.findall("visual"):
            if visual.find("geometry/mesh") is None or name in out:
                continue
            origin = visual.find("origin")
            out[name] = {
                "rpy": triple(origin.get("rpy") if origin is not None else None),
                "xyz": triple(origin.get("xyz") if origin is not None else None),
            }
    return out


pytest.importorskip("flask")

from uniurdf.gui import app as app_module  # noqa: E402
from uniurdf.gui.app import (  # noqa: E402
    Session,
    create_app,
    describe_model,
)

# Located through the import, so these tests read the assets the installed
# package serves rather than a copy in the source tree. Must sit below the
# importorskip: naming the module is what pulls Flask in.
_STATIC_DIR = Path(app_module.__file__).resolve().parent / "static"

# The vendored three.js is fetched, not committed, so a fresh clone has none
# of it until `make vendor` runs.
_vendored = pytest.mark.skipif(
    not (_STATIC_DIR / "vendor" / "three.module.min.js").is_file(),
    reason="vendored JavaScript absent; run scripts/fetch_vendor.py",
)

ARM_URDF = b"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="l1">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><mesh filename="meshes/upper.STL" scale="1 1 1"/></geometry>
    </visual>
  </link>
  <link name="l2">
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry><mesh filename="package://gripper/meshes/jaw.STL"/></geometry>
    </visual>
  </link>
  <link name="floater"/>
  <link name="finger_a"/><link name="finger_b"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 -1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="continuous">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="grip_a" type="prismatic">
    <parent link="l2"/><child link="finger_a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
  <joint name="grip_b" type="prismatic">
    <parent link="l2"/><child link="finger_b"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="-1 0 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
</robot>"""

# A URDF with a cycle so it cannot be loaded.
BROKEN_URDF = b"""<?xml version="1.0"?>
<robot name="broken">
  <link name="base"/>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="l1"/><child link="ghost_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""

SPEC = {
    "chains": [["j1", "j2"]],
    "tcp": [
        {
            "base_link": "l2",
            "xyz": [0, 0, 0.15],
            "rpy": [0, 0, 270],
            "joints": ["grip_a", "grip_b"],
        }
    ],
}


def _make_project(tmp_path: Path, stem: str = "arm") -> Project:
    """Build a minimal Project pointing at ARM_URDF."""
    from uniurdf.spec import AlignmentSpec

    source = tmp_path / "robot" / f"{stem}.urdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(ARM_URDF)
    (source.parent / "meshes").mkdir(exist_ok=True)
    (source.parent / "meshes" / "upper.STL").write_bytes(b"solid\n")
    # A `package://` ref resolves under mesh_root as <pkg>/<rel>, and is
    # copied to that same path -- so the viewer must strip the scheme to
    # find it. Without one here, the viewer's plain-path assumption held.
    jaw = source.parent / "gripper" / "meshes" / "jaw.STL"
    jaw.parent.mkdir(parents=True, exist_ok=True)
    jaw.write_bytes(b"solid jaw\n")
    (source.parent / "secret.txt").write_text("not a mesh")
    spec = parse_spec(
        {
            **SPEC,
            "source_urdf": str(source),
        },
        origin="<test>",
    )
    # Resolve source_urdf path properly
    aligned_spec = AlignmentSpec(
        source_urdf=source,
        mesh_root=source.parent,
        output=None,
        chains=spec.chains,
        tcp=spec.tcp,
        camera_references=spec.camera_references,
    )
    return Project(
        spec=aligned_spec,
        source_urdf=source,
        mesh_root=source.parent,
        output=(tmp_path / "out").resolve(),
        stem=stem,
    )


@pytest.fixture()
def session(tmp_path: Path) -> Session:
    project = _make_project(tmp_path)
    staged = project.stage()
    return Session(project=project, staged=staged)


@pytest.fixture()
def client(session: Session):
    return create_app(session).test_client()


@pytest.fixture()
def fatal_session(tmp_path: Path) -> Session:
    """A session whose staged URDF has a fatal defect (undeclared link ref)."""
    from uniurdf.assets import CopiedUrdfAsset

    stem = "broken"
    staged_path = tmp_path / "out" / f"{stem}.source.urdf"
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(BROKEN_URDF)
    diagnostics = inspect_urdf(BROKEN_URDF)
    # Build a fake CopiedUrdfAsset with the staged path.
    asset = CopiedUrdfAsset(urdf=staged_path)

    spec = AlignmentSpec()
    project = Project(
        spec=spec,
        source_urdf=staged_path,
        mesh_root=tmp_path / "out",
        output=(tmp_path / "out").resolve(),
        stem=stem,
    )

    # Robot is None: the URDF references ghost_link which is undeclared.
    staged = StagedRobot(
        staged_path=staged_path,
        robot=None,
        diagnostics=diagnostics,
        asset=asset,
    )
    return Session(project=project, staged=staged)


@pytest.fixture()
def fatal_client(fatal_session: Session):
    return create_app(fatal_session).test_client()


class TestModelPayload:
    """What the browser needs to draw the robot once."""

    def test_describes_links_edges_and_joints(self, session: Session) -> None:
        model = describe_model(session)
        assert model["robot"] == "arm"
        # Stage 1 describes the source (staged) URDF only, no TCP link yet.
        assert "base" in model["links"]
        assert "l1" in model["links"]
        assert "l2" in model["links"]
        joint_names = [j["name"] for j in model["joints"]]
        # j1 and j2 are the arm joints; grip_a / grip_b are the gripper joints
        # declared in tcp.joints.
        assert "j1" in joint_names
        assert "j2" in joint_names

    def test_edge_indices_are_in_range(self, session: Session) -> None:
        model = describe_model(session)
        for a, b in model["edges"]:
            assert 0 <= a < len(model["links"])
            assert 0 <= b < len(model["links"])

    def test_continuous_joint_gets_a_usable_range(self, session: Session) -> None:
        # Continuous joints declare no limits, but a slider still needs one.
        joint = next(j for j in describe_model(session)["joints"] if j["name"] == "j2")
        assert joint["lower"] == pytest.approx(-np.pi)
        assert joint["upper"] == pytest.approx(np.pi)

    def test_declared_limits_are_used(self, session: Session) -> None:
        joint = next(j for j in describe_model(session)["joints"] if j["name"] == "j1")
        assert (joint["lower"], joint["upper"]) == (-1.5, 1.5)

    def test_groups_joints_by_chain(self, session: Session) -> None:
        assert describe_model(session)["chains"] == [["j1", "j2"]]

    def test_reports_orphan_links_as_a_warning(self, session: Session) -> None:
        # `floater` is declared but unreachable, so it is not drawn -- say so
        # rather than leaving a silent gap.
        warnings = describe_model(session)["warnings"]
        assert any("not connected" in text for text in warnings)

    def test_spec_angles_come_back_in_degrees(self, session: Session) -> None:
        # The panel edits degrees; radians are an internal detail.
        assert describe_model(session)["spec"]["tcp"][0]["rpy"][2] == (
            pytest.approx(270.0)
        )

    def test_reports_visual_meshes(self, session: Session) -> None:
        model = describe_model(session)
        # Stage 1 shows source URDF origins (no alignment yet applied).
        meshes = model["meshes"]
        assert "l1" in meshes
        assert meshes["l1"][0]["filename"] == "meshes/upper.STL"
        # Source origin is unchanged at stage 1.
        assert meshes["l1"][0]["origin"]["xyz"] == [0.0, 0.0, 0.1]

    def test_package_refs_carry_the_path_they_were_copied_to(
        self, session: Session
    ) -> None:
        """A `package://` ref must name where its copy actually landed.

        Copying strips the scheme and mirrors the ref at `<pkg>/<rel>`, but
        the staged `.source.urdf` keeps the original string verbatim. The
        viewer reads that URDF, so treating the ref as a plain path made it
        fetch the literal directory `package:/gripper/...` -- a 404 for every
        `package://` link while the file sat on disk. On robotwin_aloha that
        was 41 of 52 meshes: the chassis drew, both arms did not.
        """
        meshes = describe_model(session)["meshes"]
        entry = meshes["l2"][0]
        # The verbatim ref is preserved -- that is what a warning must name
        # for the user to find it in the URDF.
        assert entry["filename"] == "package://gripper/meshes/jaw.STL"
        # ...and the served path is the copy layout, scheme stripped.
        assert entry["path"] == "gripper/meshes/jaw.STL"

    def test_package_ref_meshes_are_actually_servable(
        self, session: Session, client
    ) -> None:
        """The end of the chain: the path the client is handed must fetch.

        Asserting the payload alone would not have caught the bug, since the
        client rebuilt the URL itself.
        """
        entry = describe_model(session)["meshes"]["l2"][0]
        response = client.get(f"/meshes/{entry['path']}")
        assert response.status_code == 200
        assert response.data == b"solid jaw\n"

    def test_notes_summarise_what_will_be_drawn(self, session: Session) -> None:
        notes = describe_model(session)["notes"]
        assert any("links" in note and "joints" in note for note in notes)

    def test_notes_report_resolved_meshes(self, session: Session) -> None:
        assert any("all resolved" in note for note in describe_model(session)["notes"])

    def test_notes_name_unresolved_meshes(
        self, session: Session, tmp_path: Path
    ) -> None:
        """An empty viewer must explain itself.

        Pointing the mesh root at the wrong tree is the easiest mistake to
        make -- and without this the viewer just renders bare frames with no
        indication why.
        """
        # Build a project with an output dir that has no meshes.
        empty_out = tmp_path / "empty_out"
        empty_out.mkdir()
        # Stage into empty_out: meshes won't resolve from source if source
        # dir is different. Use a session with mesh_root pointing elsewhere.
        elsewhere_project = Project(
            spec=session.project.spec,
            source_urdf=session.project.source_urdf,
            mesh_root=tmp_path / "empty_mesh_root",
            output=empty_out.resolve(),
            stem=session.project.stem,
        )
        (tmp_path / "empty_mesh_root").mkdir()
        from uniurdf.assets import CopiedUrdfAsset

        staged_path = empty_out / "arm.source.urdf"
        staged_path.write_bytes(ARM_URDF)
        staged = StagedRobot(
            staged_path=staged_path,
            robot=load_robot(staged_path),
            diagnostics=inspect_urdf(ARM_URDF),
            asset=CopiedUrdfAsset(urdf=staged_path),
        )
        elsewhere = Session(project=elsewhere_project, staged=staged)
        notes = describe_model(elsewhere)["notes"]
        assert any("not found" in note for note in notes)
        # `mesh_root` is a config field; the note must name a lever the
        # reader actually has.
        assert any("mesh_root" in note for note in notes)
        assert not any("--mesh-root" in note for note in notes)
        assert any("meshes/upper.STL" in note for note in notes)

    def test_reports_the_mesh_root(self, session: Session) -> None:
        assert describe_model(session)["mesh_root"]

    def test_focus_links_are_the_ones_the_spec_selects(self, session: Session) -> None:
        """Only the frames under review should be labelled.

        A robot carries far more links than an alignment concerns itself
        with -- 34 against 16 on a dual-arm UR5 -- and drawing a name on every
        one buries the frames being checked.
        """
        model = describe_model(session)
        focus = model["focus_links"]
        # At stage 1 (source), the spec drives j1→l1, j2→l2 (no tcp yet).
        assert "l1" in focus
        assert "l2" in focus
        # The skeleton still covers every link, so nothing vanishes.
        assert set(focus) <= set(model["links"])

    def test_focus_links_include_tcp_base_link(self, session: Session) -> None:
        """TCP base_link must appear in focus_links at stage 1.

        The base link exists in the source URDF, so it is drawn with a triad
        and the user can see which link the TCP attaches to.
        """
        model = describe_model(session)
        # l2 is the TCP base link in SPEC.
        assert "l2" in model["focus_links"]

    def test_focus_links_include_declared_camera_references(
        self, session: Session
    ) -> None:
        """A camera frame is the reason someone opens the viewer.

        The pruned URDF keeps every declared `camera_references` link, so
        the viewer must label them too -- otherwise the frame under review
        is the one frame not drawn.
        """
        model = describe_model(session)
        for reference in session.spec.camera_references:
            assert reference.link in model["focus_links"]

    def test_focus_links_match_what_pruning_keeps(self, session: Session) -> None:
        """One selection rule, so the two answers cannot drift apart."""
        from uniurdf.prune import focus_links

        assert session.robot is not None
        model = describe_model(session)
        expected = focus_links(session.robot.chain, session.spec)
        assert set(model["focus_links"]) == set(expected)

    def test_focus_links_include_tcp_joint_children(self, session: Session) -> None:
        """Children of tcp.joints appear in focus_links at stage 1.

        Gripper joints are declared in the source URDF; their child links
        must be drawn so the user can see the gripper geometry.
        """
        model = describe_model(session)
        # grip_a -> finger_a, grip_b -> finger_b (from ARM_URDF).
        assert "finger_a" in model["focus_links"], (
            "child of tcp.joints[0] (grip_a -> finger_a) missing from focus_links"
        )
        assert "finger_b" in model["focus_links"], (
            "child of tcp.joints[1] (grip_b -> finger_b) missing from focus_links"
        )
        # Non-zero count guard.
        assert len(model["focus_links"]) > 0

    def test_spec_payload_includes_joints(self, session: Session) -> None:
        """tcp.joints round-trips through /api/model's spec payload."""
        spec_payload = describe_model(session)["spec"]
        tcp_entries = spec_payload["tcp"]
        assert tcp_entries, "fixture must declare a TCP"
        entry = tcp_entries[0]
        assert "joints" in entry, "joints key missing from spec payload"
        assert entry["joints"] == ["grip_a", "grip_b"]

    def test_model_payload_includes_tcp_base_links(self, session: Session) -> None:
        """model.tcp_base_links lists the TCP base link for each TCP entry."""
        model = describe_model(session)
        assert "tcp_base_links" in model
        assert model["tcp_base_links"] == ["l2"]

    def test_model_payload_includes_tcp_joints_list(self, session: Session) -> None:
        """model.tcp_joints groups gripper joint names per TCP entry."""
        model = describe_model(session)
        assert "tcp_joints" in model
        assert model["tcp_joints"] == [["grip_a", "grip_b"]]

    def test_focus_links_have_no_duplicates(self, session: Session) -> None:
        focus = describe_model(session)["focus_links"]
        assert len(focus) == len(set(focus))

    def test_diagnostics_are_included(self, session: Session) -> None:
        """The model payload must carry structural findings.

        The GUI is the tool used to repair a broken URDF (decision 17), so
        findings must be visible at stage 1 before any alignment is attempted.
        """
        model = describe_model(session)
        assert "diagnostics" in model
        # ARM_URDF has one warning (orphan link 'floater').
        assert isinstance(model["diagnostics"], list)
        codes = {d["code"] for d in model["diagnostics"]}
        assert "orphan_link" in codes

    def test_diagnostics_have_severity_code_message(self, session: Session) -> None:
        for finding in describe_model(session)["diagnostics"]:
            assert "severity" in finding
            assert "code" in finding
            assert "message" in finding

    def test_asset_report_is_included(self, session: Session) -> None:
        model = describe_model(session)
        assert "asset" in model
        asset = model["asset"]
        assert "staged_urdf" in asset
        assert "copied_mesh_count" in asset
        assert "unresolved_meshes" in asset

    def test_source_urdf_path_is_staged_path(self, session: Session) -> None:
        """The source_urdf field must point at the staged copy.

        The user edits the staged copy in place; the field must give them the
        right path to open.
        """
        model = describe_model(session)
        assert str(session.staged.staged_path) == model["source_urdf"]

    def test_mesh_origins_come_from_the_source_urdf(self, session: Session) -> None:
        """At stage 1, mesh origins and source FK must describe the same robot.

        Pairing source origins with aligned poses applies the rotation once
        instead of twice, which leaves most meshes visibly turned. Stage 1
        shows the *source* -- origins and poses must both be from the source.

        The invariant: a mesh's pose in the world is what alignment must
        preserve, since alignment changes conventions and not geometry.
        """
        model = describe_model(session)
        robot = session.robot
        assert robot is not None
        source_poses = robot.fk()
        source_visuals = _visual_origins(robot.urdf_bytes)
        assert source_visuals, "fixture must declare a visual mesh"

        for link, entries in model["meshes"].items():
            served = entries[0]["origin"]
            in_world = source_poses[link] @ _origin_matrix(served["rpy"], served["xyz"])
            original = source_visuals[link]
            expected = source_poses[link] @ _origin_matrix(
                original["rpy"], original["xyz"]
            )
            np.testing.assert_allclose(
                in_world,
                expected,
                atol=1e-9,
                err_msg=f"{link}: mesh moved in source stage",
            )

    def test_fatal_session_returns_empty_links(self, fatal_session: Session) -> None:
        """An unloadable URDF still returns a valid JSON response."""
        model = describe_model(fatal_session)
        assert model["links"] == []
        assert model["joints"] == []
        assert model["meshes"] == {}

    def test_fatal_session_reports_diagnostics(self, fatal_session: Session) -> None:
        model = describe_model(fatal_session)
        assert model["diagnostics"]
        severities = {d["severity"] for d in model["diagnostics"]}
        assert "fatal" in severities

    def test_fatal_session_notes_mention_staged_path(
        self, fatal_session: Session
    ) -> None:
        model = describe_model(fatal_session)
        combined = " ".join(model["notes"])
        assert str(fatal_session.staged.staged_path) in combined


class TestPoseEndpoint:
    """Joint values in, link transforms out."""

    def test_source_stage_returns_transforms(self, client) -> None:
        response = client.post(
            "/api/pose", json={"seq": 1, "joints": {}, "stage": "source"}
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert set(payload["transforms"]) >= {"base", "l1", "l2"}
        assert len(payload["transforms"]["base"]) == 16

    def test_aligned_stage_includes_tcp_link(self, client) -> None:
        response = client.post(
            "/api/pose", json={"seq": 1, "joints": {}, "stage": "aligned"}
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert "l2_tcp" in payload["transforms"]
        assert payload["tcp_links"] == ["l2_tcp"]

    def test_default_stage_is_source(self, client) -> None:
        # Omitting `stage` should behave like stage=source.
        r1 = client.post("/api/pose", json={"seq": 1, "joints": {}}).get_json()
        r2 = client.post(
            "/api/pose", json={"seq": 1, "joints": {}, "stage": "source"}
        ).get_json()
        assert set(r1["transforms"].keys()) == set(r2["transforms"].keys())

    def test_source_transforms_match_source_fk(self, client, session: Session) -> None:
        """Stage-source FK must equal the source robot's FK directly."""
        values = {"j1": 0.4, "j2": -0.9}
        response = client.post(
            "/api/pose",
            json={"seq": 1, "joints": values, "stage": "source"},
        )
        served = response.get_json()["transforms"]
        assert session.robot is not None
        expected = session.robot.fk(values)
        for name, matrix in expected.items():
            np.testing.assert_allclose(
                np.array(served[name]).reshape(4, 4), matrix, atol=1e-12
            )

    def test_aligned_transforms_match_aligned_fk(
        self, client, session: Session
    ) -> None:
        """Stage-aligned FK must match the aligned robot's FK."""
        values = {"j1": 0.4, "j2": -0.9}
        response = client.post(
            "/api/pose",
            json={"seq": 1, "joints": values, "stage": "aligned"},
        )
        served = response.get_json()["transforms"]
        assert session.robot is not None
        expected = session.robot.align(session.spec).fk(values)
        for name, matrix in expected.items():
            np.testing.assert_allclose(
                np.array(served[name]).reshape(4, 4), matrix, atol=1e-12
            )

    def test_echoes_the_sequence_number(self, client) -> None:
        # The client discards stale replies by sequence, so it must come back.
        assert (
            client.post("/api/pose", json={"seq": 7, "joints": {}}).get_json()["seq"]
            == 7
        )

    def test_unknown_joints_are_ignored(self, client) -> None:
        response = client.post("/api/pose", json={"seq": 1, "joints": {"ghost": 1.0}})
        assert response.status_code == 200

    def test_rejects_a_non_object_joints_field(self, client) -> None:
        response = client.post("/api/pose", json={"joints": [1, 2]})
        assert response.status_code == 400

    def test_rejects_invalid_stage(self, client) -> None:
        response = client.post("/api/pose", json={"joints": {}, "stage": "invalid"})
        assert response.status_code == 400

    def test_accepts_a_spec_from_the_request(self, client) -> None:
        response = client.post(
            "/api/pose",
            json={
                "seq": 1,
                "joints": {},
                "stage": "aligned",
                "spec": {"tcp": [{"base_link": "l1", "xyz": [0, 0, 0.2]}]},
            },
        )
        assert response.get_json()["tcp_links"] == ["l1_tcp"]

    def test_rejects_a_malformed_spec(self, client) -> None:
        response = client.post(
            "/api/pose",
            json={"joints": {}, "stage": "aligned", "spec": {"nonsense": 1}},
        )
        assert response.status_code == 400
        assert "unknown field" in response.get_json()["error"]

    def test_spec_contradicting_the_urdf_returns_json_not_a_traceback(
        self, client
    ) -> None:
        """A typo in the authoring panel is a user error, not a server error.

        Validation against the URDF happens inside `align()`, one call later
        than parsing, so handling only the parse raised a 500 HTML traceback
        the panel could not display.
        """
        response = client.post(
            "/api/pose",
            json={
                "joints": {},
                "stage": "aligned",
                "spec": {"tcp": [{"base_link": "ghost"}]},
            },
        )
        assert response.status_code == 400
        assert response.content_type.startswith("application/json")
        assert "not a link in the URDF" in response.get_json()["error"]

    def test_fatal_session_returns_422_not_500(self, fatal_client) -> None:
        """Unloadable URDF must return a structured error, not a traceback."""
        response = fatal_client.post(
            "/api/pose", json={"joints": {}, "stage": "source"}
        )
        assert response.status_code == 422
        body = response.get_json()
        assert "error" in body
        assert "diagnostics" in body

    def test_fatal_session_aligned_stage_returns_422(self, fatal_client) -> None:
        response = fatal_client.post(
            "/api/pose", json={"joints": {}, "stage": "aligned"}
        )
        assert response.status_code == 422


class TestAlignEndpoint:
    """POST /api/align: align, verify, delta table."""

    def test_returns_the_aligned_link_topology(self, client) -> None:
        """The aligned robot has links the source does not.

        The page builds its link groups from the source `model.links`, so
        without this the emitted `_tcp` poses arrive with nowhere to land and
        are dropped: `/api/pose` returned 39 transforms into 35 groups and the
        TCP triads never drew. Reported from the browser, invisible to every
        test, because nothing asserted the two link sets agree.
        """
        model = client.get("/api/model").get_json()
        body = client.post("/api/align", json={}).get_json()
        assert set(model["links"]) <= set(body["links"])
        assert len(body["links"]) > len(model["links"])

    def test_every_aligned_pose_has_a_link(self, client) -> None:
        """No transform may be returned for a link the page cannot draw."""
        body = client.post("/api/align", json={}).get_json()
        pose = client.post(
            "/api/pose", json={"joints": {}, "stage": "aligned"}
        ).get_json()
        assert pose["transforms"]
        assert not set(pose["transforms"]) - set(body["links"])

    def test_emitted_frames_are_in_the_aligned_focus(self, client) -> None:
        """TCP and compat links are labelled, being what alignment created."""
        body = client.post("/api/align", json={}).get_json()
        emitted = body["alignment"]["tcp_links"]
        emitted += body["alignment"]["compat_links"]
        assert emitted, "spec emits no frames; this test would be vacuous"
        for name in emitted:
            assert name in body["links"]
            assert name in body["focus_links"]

    def test_aligned_edges_index_the_aligned_links(self, client) -> None:
        """Skeleton indices must address the list they are drawn against."""
        body = client.post("/api/align", json={}).get_json()
        assert body["edges"]
        for pair in body["edges"]:
            for position in pair:
                assert 0 <= position < len(body["links"])

    def test_returns_aligned_mesh_origins(self, client, session) -> None:
        """Stage 2 must serve the *aligned* URDF's visual origins.

        Alignment folds the axis rotation's inverse into the visual origins
        of every link it reorients, so those origins differ from the source's.
        The page draws stage 2 with aligned FK poses; pairing them with source
        origins applies the rotation once instead of twice and leaves the mesh
        visibly turned -- 180 degrees on the piper gripper, and 237 meshes
        across 27 of the 28 shipped specs.
        """
        body = client.post("/api/align", json={}).get_json()
        assert "meshes" in body, "aligned stage serves no mesh origins"

        aligned = session.robot.align(session.spec)
        expected = _visual_origins(aligned.urdf_bytes)
        source = _visual_origins(session.robot.urdf_bytes)
        differing = [
            link
            for link in expected
            if link in source
            and not np.allclose(expected[link]["rpy"], source[link]["rpy"], atol=1e-9)
        ]
        assert differing, (
            "fixture has no compensated visual origin, so this test could "
            "not tell the two URDFs apart"
        )
        for link in differing:
            served = body["meshes"][link][0]["origin"]
            np.testing.assert_allclose(
                served["rpy"],
                expected[link]["rpy"],
                atol=1e-9,
                err_msg=f"{link}: served the source origin, not the aligned",
            )

    def test_aligned_meshes_hold_still_in_the_world(self, client, session) -> None:
        """The invariant, end to end through the API.

        Combining what the two stages actually serve -- source origins with
        source poses, aligned origins with aligned poses -- must place every
        mesh at the same world pose. This is what the user sees when they
        toggle, and it is the whole point of alignment.
        """
        body = client.post("/api/align", json={}).get_json()
        model = describe_model(session)
        aligned = session.robot.align(session.spec)
        source_poses = session.robot.fk()
        aligned_poses = aligned.fk()

        for link, entries in model["meshes"].items():
            before = source_poses[link] @ _origin_matrix(
                entries[0]["origin"]["rpy"], entries[0]["origin"]["xyz"]
            )
            served = body["meshes"][link][0]["origin"]
            after = aligned_poses[link] @ _origin_matrix(served["rpy"], served["xyz"])
            np.testing.assert_allclose(
                after,
                before,
                atol=1e-9,
                err_msg=f"{link}: mesh moved between the two stages",
            )

    def test_returns_alignment_section(self, client) -> None:
        body = client.post("/api/align", json={}).get_json()
        assert "alignment" in body
        assert "normalized_joints" in body["alignment"]
        assert "tcp_links" in body["alignment"]
        assert "compat_links" in body["alignment"]
        assert "warnings" in body["alignment"]

    def test_returns_verify_section(self, client) -> None:
        body = client.post("/api/align", json={}).get_json()
        assert "verify" in body
        v = body["verify"]
        assert "ok" in v
        assert "max_link_position_error" in v
        assert "max_mesh_position_error" in v
        assert "max_mesh_rotation_error" in v
        assert "checked_configurations" in v

    def test_verify_passes_on_well_formed_robot(self, client) -> None:
        body = client.post("/api/align", json={}).get_json()
        assert body["verify"]["ok"] is True

    def test_returns_delta_table(self, client) -> None:
        body = client.post("/api/align", json={}).get_json()
        assert "deltas" in body
        assert isinstance(body["deltas"], list)
        assert len(body["deltas"]) > 0

    def test_delta_table_has_required_fields(self, client) -> None:
        deltas = client.post("/api/align", json={}).get_json()["deltas"]
        for row in deltas:
            assert "link" in row
            assert "position_delta" in row
            assert "rotation_delta" in row
            assert "mesh_position_delta" in row
            assert "mesh_rotation_delta" in row

    def test_delta_link_position_is_near_zero(self, client) -> None:
        """Alignment must not move links."""
        deltas = client.post("/api/align", json={}).get_json()["deltas"]
        for row in deltas:
            assert row["position_delta"] == pytest.approx(0.0, abs=1e-9), (
                f"link {row['link']!r} moved by {row['position_delta']}"
            )

    def test_delta_mesh_position_is_near_zero(self, client) -> None:
        """Alignment must not move meshes in the world frame."""
        deltas = client.post("/api/align", json={}).get_json()["deltas"]
        for row in deltas:
            if row["mesh_position_delta"] is not None:
                assert row["mesh_position_delta"] == pytest.approx(0.0, abs=1e-9), (
                    f"mesh on link {row['link']!r} moved by "
                    f"{row['mesh_position_delta']}"
                )

    def test_delta_mesh_rotation_is_near_zero(self, client) -> None:
        """Alignment must not rotate meshes in the world frame."""
        deltas = client.post("/api/align", json={}).get_json()["deltas"]
        for row in deltas:
            if row["mesh_rotation_delta"] is not None:
                # Allow 1e-4 deg: decimal round-trip noise in URDF origins.
                assert row["mesh_rotation_delta"] == pytest.approx(0.0, abs=1e-4), (
                    f"mesh on link {row['link']!r} rotated by "
                    f"{row['mesh_rotation_delta']}"
                )

    def test_delta_rotation_nonzero_for_normalized_links(
        self, client, session: Session
    ) -> None:
        """Link frames must rotate when their joint axis is normalised.

        This is alignment working correctly -- the delta table is the only
        way to see it.
        """
        body = client.post("/api/align", json={}).get_json()
        normalized = set(body["alignment"]["normalized_joints"])
        assert normalized, "fixture must have a joint to normalize"
        assert session.robot is not None
        chain = session.robot.chain
        # Find the child links of normalized joints.
        normalized_children = {
            chain.child_link_of(j)
            for j in normalized
            if chain.child_link_of(j) is not None
        }
        deltas = {row["link"]: row for row in body["deltas"]}
        for link in normalized_children:
            assert link in deltas, f"link {link!r} missing from delta table"
            assert deltas[link]["rotation_delta"] > 0.01, (
                f"link {link!r} should rotate after normalization"
            )

    def test_accepts_spec_override(self, client) -> None:
        body = client.post(
            "/api/align",
            json={
                "spec": {
                    "chains": [["j1", "j2"]],
                    "tcp": [{"base_link": "l1", "xyz": [0, 0, 0.1]}],
                }
            },
        ).get_json()
        assert "l1_tcp" in body["alignment"]["tcp_links"]

    def test_rejects_malformed_spec(self, client) -> None:
        response = client.post("/api/align", json={"spec": {"bad_field": 1}})
        assert response.status_code == 400

    def test_fatal_session_returns_422(self, fatal_client) -> None:
        response = fatal_client.post("/api/align", json={})
        assert response.status_code == 422
        body = response.get_json()
        assert "error" in body
        assert "diagnostics" in body

    def test_mesh_origins_in_delta_use_each_urdf(
        self, client, session: Session
    ) -> None:
        """Delta table uses visual_world_poses with paired URDF + FK.

        Pairing source origins with aligned poses (or vice versa) is the
        exact bug that shipped once in this project. The delta table must
        use source URDF with source FK, and aligned URDF with aligned FK.

        The invariant: mesh_rotation_delta ≈ 0 even on links whose frame
        rotated (rotation_delta > 0).
        """
        body = client.post("/api/align", json={}).get_json()
        assert body["verify"]["ok"] is True
        for row in body["deltas"]:
            if row["mesh_rotation_delta"] is not None:
                # Allow 1e-4 deg for decimal round-trip noise.
                assert row["mesh_rotation_delta"] < 1e-2, (
                    f"link {row['link']!r}: mesh rotated "
                    f"{row['mesh_rotation_delta']:.4f} deg -- "
                    f"likely source/aligned URDF mismatch"
                )


class TestExportConfigEndpoint:
    """POST /api/export/config: write the spec as YAML."""

    def test_writes_yaml_file(self, client, session: Session) -> None:
        response = client.post("/api/export/config", json={})
        assert response.status_code == 200
        body = response.get_json()
        assert "config" in body
        written = Path(body["config"])
        assert written.exists()
        assert written.suffix == ".yaml"

    def test_written_yaml_is_parseable(self, client, session: Session) -> None:
        import yaml as _yaml

        client.post("/api/export/config", json={})
        stem = session.project.stem
        config_path = session.output_root / f"{stem}.yaml"
        assert config_path.exists()
        data = _yaml.safe_load(config_path.read_text())
        assert isinstance(data, dict)

    def test_written_yaml_contains_chains(self, client, session: Session) -> None:
        import yaml as _yaml

        client.post("/api/export/config", json={})
        stem = session.project.stem
        config_path = session.output_root / f"{stem}.yaml"
        data = _yaml.safe_load(config_path.read_text())
        assert "chains" in data

    def test_rpy_are_in_degrees(self, client, session: Session) -> None:
        """The YAML spec uses degrees; radians are an internal detail."""
        import yaml as _yaml

        client.post("/api/export/config", json={})
        stem = session.project.stem
        config_path = session.output_root / f"{stem}.yaml"
        data = _yaml.safe_load(config_path.read_text())
        tcp = data.get("tcp", [])
        assert tcp, "fixture spec must declare a TCP"
        rpy = tcp[0]["rpy"]
        # 270 degrees, not ~4.71 radians.
        assert max(abs(v) for v in rpy) == pytest.approx(270.0, abs=0.01)

    def test_writes_into_output_root(self, client, session: Session) -> None:
        body = client.post("/api/export/config", json={}).get_json()
        written = Path(body["config"])
        assert written.parent == session.output_root

    @pytest.mark.parametrize(
        "name", ["../escape.yaml", "/etc/cron.d/bad", "sub/dir.yaml", ".."]
    )
    def test_rejects_unsafe_names(self, client, name: str) -> None:
        response = client.post("/api/export/config", json={"name": name})
        assert response.status_code == 400
        assert "unsafe filename" in response.get_json()["error"]

    @pytest.mark.parametrize(
        ("label", "name"),
        [
            ("nul byte", "bad\x00name.yaml"),
            ("newline", "bad\nname.yaml"),
            ("tab", "bad\tname.yaml"),
            ("too long", "x" * 300 + ".yaml"),
        ],
    )
    def test_rejects_names_a_filesystem_cannot_hold(
        self, client, label: str, name: str
    ) -> None:
        """A mistyped name must still get a JSON reply the page can read.

        A NUL raises `ValueError` and a 300-character name raises `OSError`,
        both from the write itself -- far enough in that the reply would be a
        500 quoting a server directory, or the framework's HTML error page.
        """
        response = client.post("/api/export/config", json={"name": name})
        assert response.status_code == 400, label
        assert response.mimetype == "application/json", label

    def test_a_rejected_write_does_not_echo_the_server_path(
        self, client, session: Session
    ) -> None:
        response = client.post("/api/export/config", json={"name": "x" * 300 + ".yaml"})
        assert str(session.output_root) not in response.get_data(as_text=True)

    def test_accepts_custom_name(self, client, session: Session) -> None:
        body = client.post(
            "/api/export/config", json={"name": "custom.yaml"}
        ).get_json()
        assert body["config"].endswith("custom.yaml")
        assert Path(body["config"]).exists()

    def test_rejects_malformed_spec(self, client) -> None:
        response = client.post(
            "/api/export/config",
            json={"spec": {"bad_key": 1}},
        )
        assert response.status_code == 400

    def test_edited_spec_keeps_the_urdf_it_was_edited_against(
        self, client, session: Session
    ) -> None:
        """An exported config must be runnable by `uniurdf align`.

        The page posts structure alone -- chains, TCPs, cameras, rotations --
        so a config built from that and nothing else names no URDF, and the
        command line rejects it for the one field the page never sends.
        """
        import yaml as _yaml

        body = client.post(
            "/api/export/config",
            json={
                "name": "edited.yaml",
                "spec": {"chains": [["j1"]]},
            },
        ).get_json()
        data = _yaml.safe_load(Path(body["config"]).read_text())
        assert data["source_urdf"] == str(session.project.spec.source_urdf)

    @pytest.mark.parametrize(
        "name", ["arm.source.urdf", "arm.urdf", "arm.inventory.json"]
    )
    def test_refuses_to_write_yaml_over_another_kind_of_file(
        self, client, name: str
    ) -> None:
        """The staged `.source.urdf` is the file users hand-edit.

        A config export landing on it would replace a repaired URDF with
        YAML, so the target must carry a config's own extension.
        """
        response = client.post("/api/export/config", json={"name": name})
        assert response.status_code == 400
        assert ".yaml" in response.get_json()["error"]

    def test_the_staged_source_urdf_survives_an_export_aimed_at_it(
        self, client, session: Session
    ) -> None:
        staged = session.staged.staged_path
        before = staged.read_bytes()
        client.post("/api/export/config", json={"name": staged.name})
        assert staged.read_bytes() == before


class TestExportUrdfEndpoint:
    """POST /api/export/urdf: write the aligned URDF."""

    def test_writes_into_the_output_root(self, client, session: Session) -> None:
        response = client.post("/api/export/urdf", json={})
        assert response.status_code == 200
        written = Path(response.get_json()["urdf"])
        assert written.exists()
        assert written.parent == session.output_root

    def test_export_matches_the_library(
        self, client, session: Session, tmp_path: Path
    ) -> None:
        """The endpoint must export through the library, not beside it.

        Compared against the library's `write`, not its `urdf_bytes`: writing
        is where resolved `package://` refs become in-tree paths, so the
        in-memory bytes still carry the original scheme and are not what
        either path puts on disk.
        """
        client.post("/api/export/urdf", json={})
        written = (session.output_root / "arm.urdf").read_bytes()
        assert session.robot is not None

        reference_dir = tmp_path / "library_export"
        aligned = session.robot.align(session.spec)
        aligned.write(reference_dir, name="arm.urdf", prune_name="arm.pruned.urdf")
        assert written == (reference_dir / "arm.urdf").read_bytes()
        # And that really is the rewrite: no scheme survives the export.
        assert b"package://" not in written

    def test_writes_the_pruned_urdf_and_its_own_inventory(
        self, client, session: Session
    ) -> None:
        """Both files, every time.

        The pruned URDF is a second view of the same export, not a separate
        decision -- giving it its own button is how the two drift into
        disagreeing about which spec produced them.
        """
        payload = client.post("/api/export/urdf", json={}).get_json()
        pruned = Path(payload["pruned_urdf"])
        inventory = Path(payload["pruned_inventory"])
        assert pruned.exists() and pruned.parent == session.output_root
        assert inventory.exists()
        assert pruned.name == "arm.pruned.urdf"
        assert inventory.name == "arm.pruned.inventory.json"

    def test_pruned_inventory_indexes_the_pruned_file(
        self, client, session: Session
    ) -> None:
        """Not the full one -- that is the whole reason it is a second file."""
        payload = client.post("/api/export/urdf", json={}).get_json()
        inventory = json.loads(Path(payload["pruned_inventory"]).read_text())
        pruned = build_chain(Path(payload["pruned_urdf"]).read_bytes())
        assert inventory["joint_order"] == list(pruned.joint_names)
        for group in inventory["chains"]:
            for joint in group["joints"]:
                assert pruned.joint_names[joint["index"]] == joint["name"]

    def test_reports_whether_the_joint_vector_renumbered(self, client) -> None:
        """The client warns on this, so the flag has to be in the payload."""
        payload = client.post("/api/export/urdf", json={}).get_json()
        assert "pruned_renumbers" in payload
        assert isinstance(payload["dropped_movable_joints"], list)

    def test_is_idempotent(self, client, session: Session) -> None:
        client.post("/api/export/urdf", json={})
        first = (session.output_root / "arm.urdf").read_bytes()
        client.post("/api/export/urdf", json={})
        assert (session.output_root / "arm.urdf").read_bytes() == first

    @pytest.mark.parametrize(
        "name", ["../escape.urdf", "/etc/passwd", "sub/dir.urdf", ".."]
    )
    def test_rejects_names_that_escape_the_output_root(self, client, name: str) -> None:
        response = client.post("/api/export/urdf", json={"name": name})
        assert response.status_code == 400
        assert "unsafe filename" in response.get_json()["error"]

    def test_rejects_a_malformed_spec(self, client) -> None:
        response = client.post(
            "/api/export/urdf", json={"spec": {"tcp": [{"xyz": [0, 0, 1]}]}}
        )
        assert response.status_code == 400

    def test_reports_what_it_did(self, client) -> None:
        payload = client.post("/api/export/urdf", json={}).get_json()
        assert payload["tcp_links"] == ["l2_tcp"]
        assert "normalized_joints" in payload

    def test_fatal_session_returns_422(self, fatal_client) -> None:
        response = fatal_client.post("/api/export/urdf", json={})
        assert response.status_code == 422
        body = response.get_json()
        assert "error" in body
        assert "diagnostics" in body


class TestMeshRoute:
    """Serving geometry without serving everything else."""

    def test_serves_a_mesh(self, client) -> None:
        # Meshes were staged into the output dir.
        assert client.get("/meshes/meshes/upper.STL").status_code == 200

    def test_refuses_a_non_mesh_file(self, client, session: Session) -> None:
        # Write a non-mesh file into the output root to confirm refusal.
        (session.output_root / "secret.txt").write_text("not a mesh")
        assert client.get("/meshes/secret.txt").status_code == 403

    @pytest.mark.parametrize("path", ["../../../etc/passwd", "..%2f..%2fetc%2fpasswd"])
    def test_refuses_traversal(self, client, path: str) -> None:
        assert client.get(f"/meshes/{path}").status_code in (403, 404)

    def test_missing_mesh_is_not_found(self, client) -> None:
        assert client.get("/meshes/meshes/absent.STL").status_code == 404


class TestPage:
    """The page itself."""

    def test_serves_html(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert b"<canvas" in response.data

    def test_loads_no_remote_assets(self, client) -> None:
        """three.js is vendored, so the viewer must work offline."""
        body = client.get("/").data
        assert b'src="http' not in body
        assert b'href="http' not in body
        assert b"cdn." not in body

    @_vendored
    def test_serves_vendored_three_js(self, client) -> None:
        response = client.get("/static/vendor/three.module.min.js")
        assert response.status_code == 200

    @_vendored
    def test_every_vendored_relative_import_resolves(self, client) -> None:
        """A single unresolvable import aborts the whole module graph.

        ES modules fail as a unit: when `ColladaLoader` could not reach
        `../loaders/TGALoader.js`, nothing ran -- no scene, no `/api/model`
        call, just a page stuck on "loading". Only a 404 in the server log
        gave it away, so pin every relative specifier here.
        """
        sources = list(_STATIC_DIR.rglob("*.js"))
        # An empty tree would satisfy the loop below without checking
        # anything, which reads the same as a clean pass.
        assert sources, f"no JavaScript found under {_STATIC_DIR}"
        unresolved = []
        for source in sources:
            for specifier in re.findall(
                r"""from\s+['"](\.[^'"]+)['"]""", source.read_text()
            ):
                served = (source.parent / specifier).resolve()
                if not served.is_file():
                    unresolved.append(f"{source.name} -> {specifier}")
                    continue
                # It must also be reachable over the route the browser uses.
                route = "/static/" + str(served.relative_to(_STATIC_DIR))
                if client.get(route).status_code != 200:
                    unresolved.append(f"{route} not served")
        assert not unresolved, f"unresolvable imports: {unresolved}"

    def test_loader_coverage_matches_the_client(self) -> None:
        """The served formats and the drawable formats must be the same set.

        A format served without a loader renders nothing and says so only in
        the console; a loader for a format never served is dead weight. A
        subset assertion would catch neither, so this compares both ways.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        drawable = {
            suffix.lstrip(".") for suffix in re.findall(r'"(\.\w+)":\s*\(\)\s*=>', page)
        }
        assert drawable == set(app_module._MESH_SUFFIXES)

    def test_collada_correction_preserves_asset_unit_scale(self) -> None:
        """Clearing the rotation must not clear the unit scale with it.

        ColladaLoader folds the file's declared metres-per-unit into
        `scene.scale`. That is a real property of the mesh -- a model authored
        in millimetres needs it -- so a correction that reset the whole
        transform would fix the rotation and shrink or explode the geometry
        instead.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function neutralizeLoaderUpAxis") :]
        body = body[: body.index("\n}")]
        assert "r.set(0, 0, 0)" in body
        assert "scale" not in body

    def test_collada_correction_matches_the_loader_rotation(self) -> None:
        """Keyed on the rotation applied, not on the file extension.

        A `Y_UP` Collada gets no rotation from the loader, so a suffix test
        would be a no-op there by luck rather than by construction -- and it
        would silently start clearing a real transform if a future loader ever
        put one on the root. Matching ColladaLoader's exact `-PI/2` about X
        touches only what the loader itself added.

        STLLoader returns raw geometry and GLTFLoader leaves the scene alone,
        so neither can trip this.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        assert "COLLADA_UP_AXIS_FIX_X = -Math.PI / 2" in page
        body = page[page.index("function neutralizeLoaderUpAxis") :]
        body = body[: body.index("\n}")]
        assert "COLLADA_UP_AXIS_FIX_X" in body
        # No format dispatch inside the correction.
        for suffix in ('".dae"', '".stl"', '".glb"', '".obj"', '".gltf"'):
            assert suffix not in body

    def test_has_an_activity_log(self, client) -> None:
        body = client.get("/").data
        assert b'id="log"' in body
        assert b"uiLog" in body

    def test_logger_runs_outside_the_module(self) -> None:
        """The log must survive the failure it most needs to report.

        An ES module that cannot resolve an import aborts before executing a
        line, so a logger defined inside it would be gone precisely when the
        page is blank. It has to live in a classic script.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        module_at = page.index('<script type="module">')
        logger_at = page.index("window.uiLog")
        assert logger_at < module_at, (
            "uiLog must be defined before, and outside, the ES module"
        )

    def test_static_route_refuses_traversal(self, client) -> None:
        assert client.get("/static/../../../etc/passwd").status_code in (
            403,
            404,
        )

    def test_stage_toggle_bar_hidden_at_load(self) -> None:
        """Toggle only makes sense after first alignment; must start hidden."""
        page = (_STATIC_DIR / "index.html").read_text()
        # The bar must have display:none inline before the module sets it.
        bar_idx = page.index('id="stage-toggle-bar"')
        # The surrounding tag must contain display:none.
        tag_start = page.rindex("<div", 0, bar_idx)
        tag_end = page.index(">", bar_idx)
        tag = page[tag_start:tag_end]
        assert "display:none" in tag or "display: none" in tag

    def test_delta_panel_hidden_at_load(self) -> None:
        """Delta panel is shown only after alignment runs."""
        page = (_STATIC_DIR / "index.html").read_text()
        panel_idx = page.index('id="delta-panel"')
        tag_start = page.rindex("<div", 0, panel_idx)
        tag_end = page.index(">", panel_idx)
        tag = page[tag_start:tag_end]
        assert "display:none" in tag or "display: none" in tag

    def test_stale_badge_hidden_at_load(self) -> None:
        """Stale badge must not appear until after the first alignment."""
        page = (_STATIC_DIR / "index.html").read_text()
        badge_idx = page.index('id="stale-badge"')
        tag_start = page.rindex("<span", 0, badge_idx)
        tag_end = page.index(">", badge_idx)
        tag = page[tag_start:tag_end]
        assert "display:none" in tag or "display: none" in tag

    def test_mark_tcp_base_link_origins_uses_sphere(self) -> None:
        """The origin marker must be a sphere, not a reuse of the triad."""
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function markTcpBaseLinkOrigins") :]
        body = body[: body.index("\n}")]
        assert "SphereGeometry" in body, (
            "markTcpBaseLinkOrigins must use a sphere, not an arrow/triad"
        )
        # Must not call triad() — that is the emitted-link vocabulary.
        assert "triad(" not in body

    def test_mark_tcp_links_replaces_the_regular_triad(self) -> None:
        """An emphasised frame must carry one triad, not two.

        `buildLinkGroups` gives every focus link a `triad()`, and TCP and
        compat links are always in the focus set. `markTcpLinks` then adds the
        oversized `triad(2.2)` on top. Both sit at the same origin on the same
        axes, so the small one ends up buried inside the large one and renders
        as doubled arrowheads with a stub on each axis -- which defeats the
        emphasis it was meant to provide.

        The removal must be keyed on the triad tag rather than clearing the
        group, since the label sprite and the TCP origin sphere are siblings
        that must survive.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        assert "userData.triad = true" in page, (
            "triad() must tag itself so markTcpLinks can find and remove it"
        )
        body = page[page.index("function markTcpLinks") :]
        body = body[: body.index("\n}\n")]
        assert "userData.triad" in body, (
            "markTcpLinks must remove the regular triad before adding the "
            "oversized one, or both render at the same origin"
        )
        assert "group.remove(" in body
        assert "triad(2.2)" in body

    def test_mark_tcp_links_keeps_labels_and_origin_dots(self) -> None:
        """Removing the triad must not take its siblings with it.

        A group holds the triad, the link-name label sprite, any loaded mesh
        holders, and possibly the TCP base-link origin sphere. Clearing the
        group wholesale would silently drop the label and the dot.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function markTcpLinks") :]
        body = body[: body.index("\n}\n")]
        for wholesale in ("group.clear(", "children.length = 0", "= []"):
            assert wholesale not in body, (
                f"markTcpLinks uses {wholesale!r}, which also removes the "
                f"label sprite, mesh holders and the TCP origin dot"
            )

    def test_triad_checkbox_exists_and_defaults_to_on(self) -> None:
        """Per-frame triad toggles, default visible.

        `hiddenTriads` holds the *hidden* names rather than the visible ones,
        which is what makes "default on" fall out for free: a link nobody has
        touched -- including one alignment has not invented yet, like
        `*_tcp` -- has no entry and is drawn.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        assert "function triadCheckbox" in page
        body = page[page.index("function triadCheckbox") :]
        body = body[: body.index("\n}\n")]
        assert 'box.type = "checkbox"' in body
        assert "checked = !hiddenTriads.has(" in body, (
            "checkbox must default to on, i.e. checked unless explicitly hidden"
        )

    def test_triad_checkbox_is_keyed_by_the_child_link(self) -> None:
        """Triads are drawn on links, sliders are named for joints.

        A joint row's box must govern `joint.child` -- the link the joint
        drives, which is where the triad actually is. Keying it by the joint
        name would address a group that does not exist.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function jointRow") :]
        body = body[: body.index("\n}\n")]
        assert "triadCheckbox(joint.child" in body, (
            "the joint row's checkbox must be keyed by joint.child, not by "
            "the joint name -- the triad belongs to the link"
        )

    def test_hidden_triad_is_not_resurrected_by_mark_tcp_links(self) -> None:
        """The regression that would make the box appear to do nothing.

        `markTcpLinks` runs on every pose reply and adds a triad to exactly
        the emitted TCP and compat frames. Without an explicit check it would
        hand a hidden one a fresh triad the moment any slider moved.

        Its `group.children.length` guard does not cover this: a hidden TCP
        base link still carries its origin sphere, so the group is non-empty.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function markTcpLinks") :]
        body = body[: body.index("\n}\n")]
        assert "hiddenTriads.has(" in body, (
            "markTcpLinks must skip hidden links, or toggling an emitted "
            "frame off is undone by the next pose reply"
        )

    def test_triad_toggle_does_not_touch_meshes_or_fit(self) -> None:
        """The setting influences triads only.

        The handler edits the one affected group in place. Rebuilding instead
        would re-fetch every mesh in the robot, which on a large one is
        seconds of loading to add or remove three arrows. `fitView` must not
        run either, or unticking a box would move the camera.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function setTriadVisible") :]
        body = body[: body.index("\n}\n")]
        assert "hiddenTriads" in body
        assert "buildLinkGroups()" not in body, (
            "a full rebuild re-fetches every mesh; edit the one group instead"
        )
        assert "loadMeshes()" not in body, "toggling a triad must not reload meshes"
        assert "requestPose()" not in body, (
            "the group keeps its transform, so no new pose is needed"
        )
        assert "fitView()" not in body, "toggling a triad must not re-fit the camera"
        assert "show-meshes" not in body, (
            "the triad toggle must not touch the mesh control"
        )

    def test_triad_toggle_respects_the_focus_rule(self) -> None:
        """Showing a triad obeys the same eligibility the build pass uses.

        `all frames` widens which links may carry a triad; it does not grant
        one to a link outside the focus set when that switch is off. Editing
        the group directly has to re-state that rule, since it no longer goes
        through the build pass that enforces it.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function setTriadVisible") :]
        body = body[: body.index("\n}\n")]
        assert "showAllFrames" in body and "userData.focus" in body, (
            "showing a triad must check focus/showAllFrames, or an unfocused "
            "link gains one the build pass would not have drawn"
        )

    def test_triad_toggle_preserves_the_emphasised_tcp_scale(self) -> None:
        """A re-shown TCP frame comes back oversized, not regular-sized.

        `markTcpLinks` only enlarges a group it has not already marked, so a
        plain triad added here would stay small for the rest of the session.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function setTriadVisible") :]
        body = body[: body.index("\n}\n")]
        assert "userData.marked" in body, (
            "re-showing an emphasised frame must restore its larger triad"
        )

    def test_fit_view_ignores_triad_visibility(self) -> None:
        """Hiding frames must not change what the camera frames.

        `fitView` selects on `group.children.length`, which stays non-empty
        for a hidden link once its mesh holder is attached, and in any case
        must not be narrowed to visible triads -- that would make the view
        jump on every tick.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function fitView") :]
        body = body[: body.index("\n}\n")]
        assert "hiddenTriads" not in body, (
            "fitView must not consult triad visibility, or the camera moves "
            "when a box is unticked"
        )

    def test_undriven_focus_frames_get_their_own_group(self) -> None:
        """TCP and compat links have no slider, so no row would carry a box.

        They hang off *fixed* joints and are precisely the frames alignment
        creates, so leaving them permanently on would exempt the most
        interesting frames from the control.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        assert "function renderUndrivenFrames" in page
        body = page[page.index("function renderUndrivenFrames") :]
        body = body[: body.index("\n}\n")]
        assert "focus_links" in body
        assert "triadCheckbox(" in body

    def test_alignment_rerenders_the_frame_checkboxes(self) -> None:
        """`renderJoints` runs once at bootstrap; alignment adds link names.

        The emitted `*_tcp` and `*_camera_mount_compat` links enter the focus
        set only when alignment runs, so without an explicit re-render they
        would never get a checkbox.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function applyAlignedTopology") :]
        body = body[: body.index("\n}\n")]
        assert "renderJoints()" in body, (
            "aligned stage must re-render the joint panel so emitted frames "
            "get a checkbox"
        )

    def test_frame_count_counts_triads_not_children(self) -> None:
        """A hidden TCP base link still has its origin sphere.

        Counting non-empty groups would report it as drawn, so the readout
        would not move when the user hid it.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        body = page[page.index("function updateFrameCount") :]
        body = body[: body.index("\n}\n")]
        assert "userData.triad" in body, (
            "frame count must count triads, not non-empty groups"
        )

    def test_build_link_groups_calls_mark_tcp_base_link_origins(
        self,
    ) -> None:
        """``buildLinkGroups`` calls markTcpBaseLinkOrigins after building."""
        page = (_STATIC_DIR / "index.html").read_text()
        # The call must appear inside buildLinkGroups, not just defined.
        body = page[page.index("function buildLinkGroups") :]
        body = body[: body.index("\n}")]
        assert "markTcpBaseLinkOrigins" in body


class TestColladaUpAxis:
    """The geometric fact behind `neutralizeLoaderUpAxis`.

    The page-text guard above pins that the fix is present. This pins *why* it
    is right, against real assets, so the reasoning survives a reader who
    doubts it -- which is the failure mode here: the bug shipped precisely
    because a rotation applied inside vendored code looked like the mesh's own
    property.
    """

    def _dae_and_stl_pairs(self, root: Path, limit: int = 4) -> list[tuple[Path, Path]]:
        """Meshes shipped as both `.dae` and `.STL` under `root`."""
        pairs = []
        for dae in sorted(root.rglob("*.dae")):
            for twin in (dae.with_suffix(".STL"), dae.with_suffix(".stl")):
                if twin.exists():
                    pairs.append((dae, twin))
                    break
            if len(pairs) >= limit:
                break
        return pairs

    def _stl_extent(self, path: Path) -> np.ndarray:
        """Bounding-box side lengths of an STL, binary or ASCII."""
        raw = path.read_bytes()
        if raw[:5].lower() == b"solid" and b"facet" in raw[:2000]:
            values = [
                float(component)
                for triple in re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw)
                for component in triple
            ]
            vertices = np.array(values, dtype=float).reshape(-1, 3)
        else:
            count = struct.unpack("<I", raw[80:84])[0]
            vertices = np.array(
                [
                    struct.unpack("<12fH", raw[84 + i * 50 : 134 + i * 50])[3:12]
                    for i in range(count)
                ]
            ).reshape(-1, 3)
        return vertices.max(axis=0) - vertices.min(axis=0)

    def _dae_extent(self, path: Path) -> np.ndarray:
        """Bounding-box side lengths from a Collada's first position array."""
        text = path.read_text(errors="ignore")
        match = re.search(r"<float_array[^>]*>([^<]+)</float_array>", text, re.S)
        assert match is not None, f"no float_array in {path}"
        vertices = np.fromstring(match.group(1), sep=" ").reshape(-1, 3)
        return vertices.max(axis=0) - vertices.min(axis=0)

    def test_collada_twins_need_no_rotation(self, urdf_root) -> None:
        """A `Z_UP` Collada already sits in the URDF's frame.

        Where a mesh ships as both formats *and the two hold the same
        geometry*, the extents agree axis-for-axis. No loader rotates an
        `.STL`, so it is ground truth: applying `ColladaLoader`'s `-PI/2` to
        the `.dae` twin would swap its Y and Z extents and put it a quarter
        turn from a mesh known to be correct.

        Same-named files are not always the same mesh -- 8 of 20 pairs in this
        corpus differ outright, and none of those differ *by a rotation*, so
        they are unrelated geometry sharing a basename rather than evidence
        against the rule. Only genuine twins are compared, and the assertion
        is that none of them is a rotation of its partner.
        """
        pairs = self._dae_and_stl_pairs(urdf_root, limit=40)
        if not pairs:
            pytest.skip("no .dae/.STL twins in the asset tree")

        compared = 0
        for dae, stl in pairs:
            declared = re.search(
                r"<up_axis>\s*([A-Z_]+)\s*</up_axis>",
                dae.read_text(errors="ignore"),
            )
            if declared is None or declared.group(1) != "Z_UP":
                continue
            try:
                dae_extent = self._dae_extent(dae)
                stl_extent = self._stl_extent(stl)
            except (AssertionError, ValueError, struct.error):
                continue
            if not np.allclose(dae_extent, stl_extent, atol=1e-4):
                # Different geometry under a shared name. It only counts
                # against the rule if it is a *rotation* of its partner.
                assert not np.allclose(
                    sorted(dae_extent), sorted(stl_extent), atol=1e-4
                ), (
                    f"{dae.name}: extents are a permutation of its STL twin's "
                    f"({dae_extent} vs {stl_extent}) -- the Collada is being "
                    f"rotated relative to a mesh no loader rotates."
                )
                continue
            compared += 1

        assert compared >= 5, (
            f"only {compared} genuine Z_UP twins compared; the corpus that "
            f"proved this rule has 12. A near-zero count means the pairing "
            f"broke, not that the rule holds."
        )


class TestStageTopologySwap:
    """The page must rebuild its link groups when the stage changes."""

    def test_page_swaps_topology_on_align(self) -> None:
        """Aligned links/edges/focus replace the source set, and back.

        Keeping the source topology after aligning is what hid the emitted
        `_tcp` frames: `buildLinkGroups` iterates `model.links`, so a link
        absent from that list gets no group, no triad, and no label however
        many transforms the server sends for it.
        """
        page = (_STATIC_DIR / "index.html").read_text()
        assert "function applyAlignedTopology" in page
        assert "function applySourceTopology" in page
        # Both must run, not merely exist.
        assert "applyAlignedTopology(payload)" in page
        assert "applySourceTopology()" in page
        # The swap has to rebuild; assigning the arrays alone draws nothing.
        body = page[page.index("function applyAlignedTopology") :]
        body = body[: body.index("\n}")]
        assert "buildLinkGroups()" in body

    def test_tcp_origin_marker_is_small(self) -> None:
        """The base-link dot marks an origin; it must not cover the triad."""
        page = (_STATIC_DIR / "index.html").read_text()
        match = re.search(r"SphereGeometry\(AXIS_LEN \* ([0-9.]+)", page)
        assert match is not None
        assert float(match.group(1)) <= 0.15
