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

"""Tests for the alignment spec.

The spec is the tool's whole user interface, so these tests lean on error
messages as much as on parsing: a spec that fails should say what to write
instead.
"""

import math
from pathlib import Path

import pytest

from uniurdf.kinematics.chain import build_chain
from uniurdf.spec import (
    AlignmentSpec,
    CameraReference,
    SpecError,
    TcpSpec,
    load_spec,
    parse_spec,
)

URDF = b"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="fixed_tip"/>
  <link name="finger_a"/><link name="finger_b"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="tip_joint" type="fixed">
    <parent link="l2"/><child link="fixed_tip"/>
  </joint>
  <joint name="grip_a" type="prismatic">
    <parent link="l2"/><child link="finger_a"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
  <joint name="grip_b" type="prismatic">
    <parent link="l2"/><child link="finger_b"/><axis xyz="-1 0 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
</robot>"""


@pytest.fixture()
def chain():
    return build_chain(URDF)


class TestParseSpec:
    """Structural parsing, before any URDF is involved."""

    def test_empty_spec_is_valid(self) -> None:
        spec = parse_spec({})
        assert spec.chains == ()
        assert spec.tcp == ()

    def test_parses_chains_as_list_of_lists(self) -> None:
        spec = parse_spec({"chains": [["a", "b"], ["c"]]})
        assert spec.chains == (("a", "b"), ("c",))

    def test_parses_tcp_with_degrees_converted_to_radians(self) -> None:
        spec = parse_spec(
            {"tcp": [{"base_link": "l2", "xyz": [0, 0, 0.2], "rpy": [0, 0, 90]}]}
        )
        tcp = spec.tcp[0]
        assert tcp.base_link == "l2"
        assert tcp.xyz == (0.0, 0.0, 0.2)
        assert tcp.rpy[2] == pytest.approx(math.pi / 2)

    def test_null_tcp_entry_means_no_tcp_for_that_chain(self) -> None:
        spec = parse_spec(
            {"chains": [["a"], ["b"]], "tcp": [None, {"base_link": "l2"}]}
        )
        assert spec.tcp[0] is None
        assert len(spec.tcps) == 1

    def test_emitted_names_derive_from_base_link(self) -> None:
        tcp = TcpSpec(base_link="fl_link6")
        assert tcp.link_name == "fl_link6_tcp"
        assert tcp.joint_name == "fl_link6_tcp_joint"

    def test_camera_reference_accepts_bare_string(self) -> None:
        spec = parse_spec({"camera_references": ["fl_link6"]})
        assert spec.camera_references == (CameraReference(link="fl_link6"),)

    def test_camera_reference_accepts_mapping_form(self) -> None:
        spec = parse_spec({"camera_references": [{"link": "fl_link6"}]})
        assert spec.camera_references == (CameraReference(link="fl_link6"),)
        assert (
            spec.camera_references[0].compat_link_name == "fl_link6_camera_mount_compat"
        )

    def test_camera_reference_rejects_insert_before(self) -> None:
        """The field was removed; a stale spec must fail loudly.

        Silently ignoring it would leave the author believing they still
        control the compat joint's position.
        """
        with pytest.raises(SpecError, match="unknown field"):
            parse_spec(
                {"camera_references": [{"link": "fl_link6", "insert_before": "j2"}]}
            )

    def test_parses_tcp_joints_as_tuple(self) -> None:
        spec = parse_spec(
            {
                "tcp": [
                    {
                        "base_link": "l2",
                        "joints": ["grip_a", "grip_b"],
                    }
                ]
            }
        )
        assert spec.tcp[0].joints == ("grip_a", "grip_b")

    def test_tcp_joints_defaults_to_empty(self) -> None:
        spec = parse_spec({"tcp": [{"base_link": "l2"}]})
        assert spec.tcp[0].joints == ()

    def test_tcp_joints_missing_field_is_silent_typo_guard(self) -> None:
        # A typo like `joint` instead of `joints` must not pass silently.
        with pytest.raises(SpecError, match="unknown field"):
            parse_spec({"tcp": [{"base_link": "l2", "joint": ["grip_a"]}]})


class TestParseErrors:
    """A malformed spec must fail loudly, naming the offending entry."""

    def test_unknown_top_level_field(self) -> None:
        # Silently ignoring a typo would emit a valid-looking, wrong URDF.
        with pytest.raises(SpecError, match="unknown field"):
            parse_spec({"chian": []})

    def test_unknown_tcp_field(self) -> None:
        with pytest.raises(SpecError, match=r"tcp\[0\] has unknown field"):
            parse_spec({"tcp": [{"base_link": "l2", "rotate_z_deg": 90}]})

    def test_flat_chain_list_is_rejected(self) -> None:
        with pytest.raises(SpecError, match="must be a list of joint names"):
            parse_spec({"chains": ["j1", "j2"]})

    def test_tcp_requires_base_link(self) -> None:
        with pytest.raises(SpecError, match="requires `base_link`"):
            parse_spec({"tcp": [{"xyz": [0, 0, 1]}]})

    def test_chain_tcp_length_mismatch(self) -> None:
        with pytest.raises(SpecError, match="paired positionally"):
            parse_spec({"chains": [["a"], ["b"]], "tcp": [{"base_link": "x"}]})

    def test_vector_must_have_three_components(self) -> None:
        with pytest.raises(SpecError, match="exactly 3 entries"):
            parse_spec({"tcp": [{"base_link": "l2", "xyz": [0, 1]}]})

    def test_vector_rejects_booleans(self) -> None:
        with pytest.raises(SpecError, match="must contain numbers"):
            parse_spec({"tcp": [{"base_link": "l2", "xyz": [0, 0, True]}]})

    def test_tcp_joints_must_be_a_list(self) -> None:
        with pytest.raises(SpecError, match="must be a list"):
            parse_spec({"tcp": [{"base_link": "l2", "joints": "grip_a"}]})

    def test_tcp_joints_entry_must_be_non_empty_string(self) -> None:
        with pytest.raises(SpecError, match="non-empty joint name"):
            parse_spec({"tcp": [{"base_link": "l2", "joints": [""]}]})


class TestValidateAgainstUrdf:
    """Errors that need the URDF to detect."""

    def test_accepts_a_coherent_spec(self, chain) -> None:
        parse_spec(
            {
                "chains": [["j1", "j2"]],
                "tcp": [{"base_link": "l2", "xyz": [0, 0, 0.1]}],
                "camera_references": [{"link": "l1"}],
            }
        ).validate_against(chain)

    def test_rejects_unknown_joint(self, chain) -> None:
        with pytest.raises(SpecError, match="does not declare as movable"):
            parse_spec({"chains": [["nope"]]}).validate_against(chain)

    def test_rejects_fixed_joint_in_chain(self, chain) -> None:
        # A fixed joint has no degree of freedom, so it cannot be selected.
        with pytest.raises(SpecError, match="does not declare as movable"):
            parse_spec({"chains": [["tip_joint"]]}).validate_against(chain)

    def test_rejects_joint_claimed_by_two_chains(self, chain) -> None:
        with pytest.raises(SpecError, match="repeats joint"):
            parse_spec(
                {"chains": [["j1"], ["j1"]], "tcp": [None, None]}
            ).validate_against(chain)

    def test_rejects_unknown_tcp_base_link(self, chain) -> None:
        with pytest.raises(SpecError, match="not a link in the URDF"):
            parse_spec({"tcp": [{"base_link": "ghost"}]}).validate_against(chain)

    def test_rejects_tcp_on_a_link_unreachable_from_the_root(self) -> None:
        """An unplaceable TCP would be reported as a clean success.

        Forward kinematics covers links below the root, so a TCP hung off a
        detached link is never placed: nothing compares it, and verification
        passes having checked nothing about the link the user asked for.
        """
        detached = build_chain(
            b"""<?xml version="1.0"?>
<robot name="detached">
  <link name="base"/><link name="l1"/><link name="lonely"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
  </joint>
</robot>"""
        )
        assert "lonely" in detached.orphan_links
        with pytest.raises(SpecError, match="unreachable from root"):
            parse_spec({"tcp": [{"base_link": "lonely"}]}).validate_against(detached)

    def test_rejects_unknown_camera_reference(self, chain) -> None:
        with pytest.raises(SpecError, match="not a link in the URDF"):
            parse_spec({"camera_references": ["ghost"]}).validate_against(chain)

    def test_default_chain_is_every_independent_joint(self, chain) -> None:
        resolved = AlignmentSpec().resolved_chains(chain)
        assert resolved == (("j1", "j2", "grip_a", "grip_b"),)

    def test_accepts_tcp_joints(self, chain) -> None:
        """Valid gripper joints in tcp.joints pass validation."""
        parse_spec(
            {
                "chains": [["j1", "j2"]],
                "tcp": [
                    {
                        "base_link": "l2",
                        "joints": ["grip_a", "grip_b"],
                    }
                ],
            }
        ).validate_against(chain)

    def test_rejects_unknown_tcp_joint(self, chain) -> None:
        with pytest.raises(SpecError, match="does not declare as movable"):
            parse_spec(
                {"tcp": [{"base_link": "l2", "joints": ["ghost_joint"]}]}
            ).validate_against(chain)

    def test_rejects_fixed_joint_in_tcp_joints(self, chain) -> None:
        # A fixed joint has no degree of freedom; it cannot be a gripper joint.
        with pytest.raises(SpecError, match="does not declare as movable"):
            parse_spec(
                {"tcp": [{"base_link": "l2", "joints": ["tip_joint"]}]}
            ).validate_against(chain)

    def test_rejects_tcp_joint_already_in_chain(self, chain) -> None:
        # A joint may not appear in both a chain and a tcp.joints list.
        # Such overlap is ambiguous about which group "owns" the joint.
        with pytest.raises(SpecError, match="already claimed by chains"):
            parse_spec(
                {
                    "chains": [["j1", "j2"]],
                    "tcp": [{"base_link": "l2", "joints": ["j1"]}],
                }
            ).validate_against(chain)

    def test_tcp_joints_do_not_change_output_urdf(self, chain) -> None:
        """The `joints` field must not alter the URDF output.

        Alignment normalises every actuated joint regardless of which joints
        are listed in tcp.joints -- the field is display metadata, not an
        export filter. Verify by aligning with and without joints declared
        and asserting byte-identical output.
        """
        from uniurdf.robot import load_robot

        robot = load_robot(URDF)
        spec_with = parse_spec(
            {
                "chains": [["j1", "j2"]],
                "tcp": [
                    {
                        "base_link": "l2",
                        "joints": ["grip_a", "grip_b"],
                    }
                ],
            }
        )
        spec_without = parse_spec(
            {
                "chains": [["j1", "j2"]],
                "tcp": [{"base_link": "l2"}],
            }
        )
        bytes_with = robot.align(spec_with).urdf_bytes
        bytes_without = robot.align(spec_without).urdf_bytes
        assert bytes_with == bytes_without, (
            "tcp.joints changed the exported URDF -- it must be visualisation-only"
        )


class TestLoadSpec:
    """File loading, including path resolution."""

    def test_resolves_source_urdf_relative_to_spec(self, tmp_path: Path) -> None:
        (tmp_path / "robots").mkdir()
        urdf = tmp_path / "robots" / "arm.urdf"
        urdf.write_bytes(URDF)
        spec_file = tmp_path / "arm.yaml"
        spec_file.write_text("source_urdf: robots/arm.urdf\n")
        # Anchoring to the spec file lets a spec and its robot move together.
        assert load_spec(spec_file).source_urdf == urdf.resolve()

    def test_keeps_absolute_source_urdf(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "arm.yaml"
        spec_file.write_text("source_urdf: /abs/robot.urdf\n")
        assert load_spec(spec_file).source_urdf == Path("/abs/robot.urdf")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="not found"):
            load_spec(tmp_path / "absent.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text("chains: [[a]\n")
        with pytest.raises(SpecError, match="not valid YAML"):
            load_spec(spec_file)

    def test_a_directory_is_reported_as_one(self, tmp_path: Path) -> None:
        """Naming a directory is an ordinary slip, and reads as one."""
        with pytest.raises(SpecError, match="is a directory"):
            load_spec(tmp_path)

    def test_a_binary_file_is_reported_as_not_text(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "binary.yaml"
        spec_file.write_bytes(bytes(range(256)))
        with pytest.raises(SpecError, match="is not text"):
            load_spec(spec_file)

    def test_empty_file_is_an_empty_spec(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "empty.yaml"
        spec_file.write_text("")
        assert load_spec(spec_file).chains == ()

    def test_parses_mesh_root_as_path(self) -> None:
        spec = parse_spec({"mesh_root": "/some/mesh/dir"})
        assert spec.mesh_root == Path("/some/mesh/dir")

    def test_parses_output_as_path(self) -> None:
        spec = parse_spec({"output": "./workspace/aligned_urdf/robotwin/ur5"})
        assert spec.output == Path("./workspace/aligned_urdf/robotwin/ur5")

    def test_resolves_mesh_root_relative_to_spec(self, tmp_path: Path) -> None:
        (tmp_path / "meshes").mkdir()
        spec_file = tmp_path / "arm.yaml"
        spec_file.write_text("mesh_root: meshes\n")
        resolved = load_spec(spec_file).mesh_root
        assert resolved == (tmp_path / "meshes").resolve()

    def test_keeps_absolute_mesh_root(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "arm.yaml"
        spec_file.write_text("mesh_root: /abs/meshes\n")
        assert load_spec(spec_file).mesh_root == Path("/abs/meshes")

    def test_output_is_not_resolved_at_load_time(self, tmp_path: Path) -> None:
        # output is CWD-relative, so load_spec must NOT resolve it.
        spec_file = tmp_path / "arm.yaml"
        spec_file.write_text("output: ./workspace/aligned_urdf/myrobot\n")
        result = load_spec(spec_file)
        assert result.output == Path("./workspace/aligned_urdf/myrobot")


class TestResolvedMeshRoot:
    """resolved_mesh_root accessor returns the effective mesh root."""

    def test_returns_explicit_mesh_root_when_set(self) -> None:
        spec = AlignmentSpec(
            source_urdf=Path("/robots/arm.urdf"),
            mesh_root=Path("/shared/meshes"),
        )
        assert spec.resolved_mesh_root == Path("/shared/meshes")

    def test_falls_back_to_source_urdf_parent(self) -> None:
        spec = AlignmentSpec(source_urdf=Path("/robots/arm.urdf"))
        assert spec.resolved_mesh_root == Path("/robots")

    def test_returns_none_when_both_unset(self) -> None:
        spec = AlignmentSpec()
        assert spec.resolved_mesh_root is None
