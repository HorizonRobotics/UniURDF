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

"""Tests for uniurdf.diagnostics.

Each test exercises exactly one defect class using a small inline URDF
string. This keeps the fixtures hermetic and the failure messages
unambiguous: when a test fails, you know which check is broken without
decoding what a 300-joint URDF contains.

The test for zero findings uses SIMPLE_URDF from conftest.py to prove the
baseline stays clean under normal conditions.
"""

from __future__ import annotations

import time

import pytest

from conftest import SIMPLE_URDF
from uniurdf.diagnostics import Diagnostics, Finding, inspect_urdf
from uniurdf.kinematics.chain import build_chain

# ---------------------------------------------------------------------------
# Inline fixture URDFs, one per defect class.
# ---------------------------------------------------------------------------

# 1. Conflicting parents: link "shared" is a child of both j1 and j2.
_CONFLICTING_PARENTS_URDF = b"""<?xml version="1.0"?>
<robot name="conflicting">
  <link name="base"/>
  <link name="shared"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="shared"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="base"/>
    <child link="shared"/>
  </joint>
</robot>
"""

# 2. Orphan: "floater" has a parent in "detached_joint", but "floater"
#    is itself never reached from the tree root (base -> attached only).
_ORPHAN_URDF = b"""<?xml version="1.0"?>
<robot name="orphan">
  <link name="base"/>
  <link name="attached"/>
  <link name="floater"/>
  <link name="floater_child"/>
  <joint name="real_joint" type="fixed">
    <parent link="base"/>
    <child link="attached"/>
  </joint>
  <joint name="detached_joint" type="revolute">
    <parent link="floater"/>
    <child link="floater_child"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# 3. Undeclared link reference: joint "bad_joint" points at "ghost",
#    which is not declared anywhere in the URDF.
_UNDECLARED_LINK_URDF = b"""<?xml version="1.0"?>
<robot name="undeclared">
  <link name="base"/>
  <link name="arm"/>
  <joint name="good_joint" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="bad_joint" type="fixed">
    <parent link="arm"/>
    <child link="ghost"/>
  </joint>
</robot>
"""

# 4. Multiple roots: both "base_a" and "base_b" have no parent joint.
_MULTIPLE_ROOTS_URDF = b"""<?xml version="1.0"?>
<robot name="multi_root">
  <link name="base_a"/>
  <link name="base_b"/>
  <link name="child"/>
  <joint name="j1" type="fixed">
    <parent link="base_a"/>
    <child link="child"/>
  </joint>
</robot>
"""

# 5. Cycle: every link is some joint's child, so no root exists.
_CYCLE_URDF = b"""<?xml version="1.0"?>
<robot name="cyclic">
  <link name="a"/>
  <link name="b"/>
  <joint name="ab" type="fixed">
    <parent link="a"/>
    <child link="b"/>
  </joint>
  <joint name="ba" type="fixed">
    <parent link="b"/>
    <child link="a"/>
  </joint>
</robot>
"""

# 6a. Duplicate joint names: "shoulder" declared twice.
_DUPLICATE_JOINT_URDF = b"""<?xml version="1.0"?>
<robot name="dup_joint">
  <link name="base"/>
  <link name="upper"/>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="shoulder" type="revolute">
    <parent link="upper"/>
    <child link="tool"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# 6b. Duplicate link names: "arm" declared twice.
_DUPLICATE_LINK_URDF = b"""<?xml version="1.0"?>
<robot name="dup_link">
  <link name="base"/>
  <link name="arm"/>
  <link name="arm"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="arm"/>
  </joint>
</robot>
"""

# 7a. Revolute joint with no <limit> element.
_NO_LIMIT_URDF = b"""<?xml version="1.0"?>
<robot name="no_limit">
  <link name="base"/>
  <link name="arm"/>
  <joint name="elbow" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

# 7b. Revolute joint where lower >= upper.
_INVERTED_LIMIT_URDF = b"""<?xml version="1.0"?>
<robot name="inverted_limit">
  <link name="base"/>
  <link name="arm"/>
  <joint name="elbow" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="1.0" upper="0.5" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# 8. Mimic joint referencing a non-existent source.
_MIMIC_MISSING_URDF = b"""<?xml version="1.0"?>
<robot name="bad_mimic">
  <link name="base"/>
  <link name="finger_a"/>
  <link name="finger_b"/>
  <joint name="drive" type="revolute">
    <parent link="base"/>
    <child link="finger_a"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="follow" type="revolute">
    <parent link="base"/>
    <child link="finger_b"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
    <mimic joint="nonexistent_joint" multiplier="1.0" offset="0.0"/>
  </joint>
</robot>
"""

# 9a. Non-unit axis vector (2-vector norm, not 1).
_NON_UNIT_AXIS_URDF = b"""<?xml version="1.0"?>
<robot name="non_unit">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="1 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# 9b. Zero-length axis vector.
_ZERO_AXIS_URDF = b"""<?xml version="1.0"?>
<robot name="zero_axis">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# 10. Unexpanded xacro expressions.
_XACRO_URDF = b"""<?xml version="1.0"?>
<robot name="xacro_unexp">
  <link name="base"/>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <origin xyz="${base_offset} 0 0" rpy="0 0 0"/>
    <limit lower="$(find pkg)/limits.yaml" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# Clean URDF that should produce zero findings.
_CLEAN_URDF = SIMPLE_URDF.encode()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _codes(diag: Diagnostics) -> list[str]:
    """Collect finding codes for concise assertions."""
    return [f.code for f in diag.findings]


# ---------------------------------------------------------------------------
# Zero-findings baseline
# ---------------------------------------------------------------------------


def test_clean_urdf_has_no_findings() -> None:
    """A well-formed URDF produces an empty Diagnostics."""
    diag = inspect_urdf(_CLEAN_URDF)
    assert diag.findings == ()
    assert not diag.is_fatal
    assert diag.fatal == ()
    assert diag.warnings == ()


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def test_severity_rule_only_cycles_and_undeclared_refs_are_fatal() -> None:
    """Only cycle and undeclared_link_reference have severity 'fatal'.

    Every other code must be a warning. This test pins the contract so that
    future code additions don't accidentally escalate something to fatal.
    """
    fatal_codes = {"cycle", "undeclared_link_reference", "xml_parse_error"}
    # Build one diagnostics object that contains all codes by combining checks.
    all_urdfs = [
        _CONFLICTING_PARENTS_URDF,
        _ORPHAN_URDF,
        _UNDECLARED_LINK_URDF,
        _MULTIPLE_ROOTS_URDF,
        _CYCLE_URDF,
        _DUPLICATE_JOINT_URDF,
        _DUPLICATE_LINK_URDF,
        _NO_LIMIT_URDF,
        _INVERTED_LIMIT_URDF,
        _MIMIC_MISSING_URDF,
        _NON_UNIT_AXIS_URDF,
        _ZERO_AXIS_URDF,
        _XACRO_URDF,
    ]
    for urdf in all_urdfs:
        diag = inspect_urdf(urdf)
        for finding in diag.findings:
            if finding.code in fatal_codes:
                assert finding.severity == "fatal", (
                    f"code '{finding.code}' expected fatal, got '{finding.severity}'"
                )
            else:
                assert finding.severity == "warning", (
                    f"code '{finding.code}' expected warning, got "
                    f"'{finding.severity}'; only cycles and "
                    f"undeclared_link_reference may be fatal"
                )


# ---------------------------------------------------------------------------
# Defect class 1: conflicting parents
# ---------------------------------------------------------------------------


def test_conflicting_parents_detected() -> None:
    diag = inspect_urdf(_CONFLICTING_PARENTS_URDF)
    assert "conflicting_parents" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "conflicting_parents")
    assert "shared" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 2: orphan links
# ---------------------------------------------------------------------------


def test_orphan_links_detected() -> None:
    diag = inspect_urdf(_ORPHAN_URDF)
    assert "orphan_link" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "orphan_link")
    # "floater" has no parent joint, but it is not the tree root ("base" is
    # declared first). "floater" and "floater_child" are both unreachable.
    assert "floater" in match.names
    assert "floater_child" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 3: undeclared link reference
# ---------------------------------------------------------------------------


def test_undeclared_link_reference_detected() -> None:
    diag = inspect_urdf(_UNDECLARED_LINK_URDF)
    assert "undeclared_link_reference" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "undeclared_link_reference")
    assert "bad_joint" in match.names
    assert "ghost" in match.names
    assert match.severity == "fatal"


def test_undeclared_link_reference_names_joint_and_missing_link() -> None:
    """Both the joint name and the undeclared link name appear in names."""
    diag = inspect_urdf(_UNDECLARED_LINK_URDF)
    match = next(f for f in diag.findings if f.code == "undeclared_link_reference")
    assert "bad_joint" in match.names
    assert "ghost" in match.names


# ---------------------------------------------------------------------------
# Defect class 4: multiple roots
# ---------------------------------------------------------------------------


def test_multiple_roots_detected() -> None:
    diag = inspect_urdf(_MULTIPLE_ROOTS_URDF)
    assert "multiple_roots" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "multiple_roots")
    assert "base_a" in match.names
    assert "base_b" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 5: cycles
# ---------------------------------------------------------------------------


def test_cycle_detected() -> None:
    diag = inspect_urdf(_CYCLE_URDF)
    assert "cycle" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "cycle")
    assert match.severity == "fatal"


def test_cycle_sets_is_fatal() -> None:
    diag = inspect_urdf(_CYCLE_URDF)
    assert diag.is_fatal


def test_cycle_below_a_valid_root_is_fatal() -> None:
    """A cycle below a valid root still leaves the URDF unloadable.

    Every link stays reachable here, so a walk that only tracks what it has
    visited sees nothing wrong. ``build_chain`` refuses to load it, and a
    merely-suspicious verdict would let the GUI call a URDF fine that cannot
    be aligned.
    """
    urdf = b"""<?xml version="1.0"?>
<robot name="back_edge">
  <link name="base"/><link name="a"/><link name="b"/>
  <joint name="j0" type="fixed">
    <parent link="base"/><child link="a"/></joint>
  <joint name="j1" type="fixed">
    <parent link="a"/><child link="b"/></joint>
  <joint name="j2" type="fixed">
    <parent link="b"/><child link="a"/></joint>
</robot>"""
    diag = inspect_urdf(urdf)
    assert diag.is_fatal
    match = next(f for f in diag.findings if f.code == "cycle")
    assert set(match.names) == {"a", "b"}

    with pytest.raises(ValueError, match="has a cycle"):
        build_chain(urdf)


def test_a_link_reachable_many_ways_is_inspected_promptly() -> None:
    """Inspection cost follows the graph's size, not its path count.

    A link reachable down several branches multiplies the number of distinct
    root-to-link walks with depth. Enumerating those walks would leave the
    CLI and the viewer's /api/model hanging on a URDF that is merely
    unusual, so each link is expanded once.
    """
    parts = ['<?xml version="1.0"?>', '<robot name="d">', '<link name="L0"/>']
    for i in range(22):
        parts += [
            f'<link name="A{i}"/>',
            f'<link name="B{i}"/>',
            f'<link name="L{i + 1}"/>',
        ]
        for child, parent in ((f"A{i}", f"L{i}"), (f"B{i}", f"L{i}")):
            parts.append(
                f'<joint name="j_{child}" type="fixed">'
                f'<parent link="{parent}"/><child link="{child}"/></joint>'
            )
        for parent in (f"A{i}", f"B{i}"):
            parts.append(
                f'<joint name="j_{parent}_L{i + 1}" type="fixed">'
                f'<parent link="{parent}"/><child link="L{i + 1}"/></joint>'
            )
    parts.append("</robot>")

    started = time.perf_counter()
    diag = inspect_urdf("".join(parts).encode())
    elapsed = time.perf_counter() - started

    assert not diag.is_fatal
    # Enumerating walks takes seconds by 20 levels and doubles every level
    # after; expanding each link once stays in milliseconds.
    assert elapsed < 2.0, f"inspect_urdf took {elapsed:.2f}s"


def test_a_urdf_with_no_links_is_not_a_cycle() -> None:
    """`build_chain` names the empty declaration, which is the actionable one.

    Asking whether any link is free of parents is vacuously false with no
    links at all, and a reader sent looking for a loop finds no joints to
    inspect.
    """
    diag = inspect_urdf(b'<robot name="empty"/>')
    assert not [f for f in diag.findings if f.code == "cycle"]


def test_a_second_root_is_not_a_cycle() -> None:
    """A stray unattached link must not make a sound tree look cyclic.

    Cycle detection walks from every root. Walking from one of several
    leaves the other trees' links unreachable, which is indistinguishable
    from a loop by reachability alone -- and a false fatal stops alignment
    on a robot that is fine.
    """
    urdf = b"""<?xml version="1.0"?>
<robot name="two_roots">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <link name="floater"/>
  <joint name="j1" type="fixed">
    <parent link="base"/><child link="l1"/></joint>
  <joint name="j2" type="fixed">
    <parent link="l1"/><child link="l2"/></joint>
</robot>"""
    diag = inspect_urdf(urdf)

    assert not diag.is_fatal
    assert not [f for f in diag.findings if f.code == "cycle"]
    # The stray link is still worth saying, just not fatally.
    assert {f.code for f in diag.findings} == {"multiple_roots", "orphan_link"}


def test_a_loop_in_a_detached_component_is_still_fatal() -> None:
    """Walking every root must not excuse a loop none of them reach."""
    urdf = b"""<?xml version="1.0"?>
<robot name="detached_loop">
  <link name="base"/><link name="x"/><link name="y"/>
  <joint name="xy" type="fixed">
    <parent link="x"/><child link="y"/></joint>
  <joint name="yx" type="fixed">
    <parent link="y"/><child link="x"/></joint>
</robot>"""
    diag = inspect_urdf(urdf)
    assert diag.is_fatal
    assert [f for f in diag.findings if f.code == "cycle"]


# ---------------------------------------------------------------------------
# Defect class 6: duplicate names
# ---------------------------------------------------------------------------


def test_duplicate_joint_name_detected() -> None:
    diag = inspect_urdf(_DUPLICATE_JOINT_URDF)
    assert "duplicate_joint_name" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "duplicate_joint_name")
    assert "shoulder" in match.names
    assert match.severity == "warning"


def test_duplicate_link_name_detected() -> None:
    diag = inspect_urdf(_DUPLICATE_LINK_URDF)
    assert "duplicate_link_name" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "duplicate_link_name")
    assert "arm" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 7: revolute limits
# ---------------------------------------------------------------------------


def test_missing_revolute_limit_detected() -> None:
    diag = inspect_urdf(_NO_LIMIT_URDF)
    assert "missing_revolute_limit" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "missing_revolute_limit")
    assert "elbow" in match.names
    assert match.severity == "warning"


def test_inverted_revolute_limit_detected() -> None:
    diag = inspect_urdf(_INVERTED_LIMIT_URDF)
    assert "inverted_revolute_limit" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "inverted_revolute_limit")
    assert "elbow" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 8: mimic targets
# ---------------------------------------------------------------------------


def test_mimic_missing_source_detected() -> None:
    diag = inspect_urdf(_MIMIC_MISSING_URDF)
    assert "mimic_missing_source" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "mimic_missing_source")
    assert "follow" in match.names
    assert "nonexistent_joint" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 9: axis vectors
# ---------------------------------------------------------------------------


def test_non_unit_axis_vector_detected() -> None:
    diag = inspect_urdf(_NON_UNIT_AXIS_URDF)
    assert "non_unit_axis_vector" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "non_unit_axis_vector")
    assert "shoulder" in match.names
    assert match.severity == "warning"


def test_zero_axis_vector_detected() -> None:
    diag = inspect_urdf(_ZERO_AXIS_URDF)
    assert "zero_axis_vector" in _codes(diag)
    match = next(f for f in diag.findings if f.code == "zero_axis_vector")
    assert "shoulder" in match.names
    assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Defect class 10: unexpanded xacro
# ---------------------------------------------------------------------------


def test_unexpanded_xacro_detected() -> None:
    diag = inspect_urdf(_XACRO_URDF)
    assert "unexpanded_xacro" in _codes(diag)
    matches = [f for f in diag.findings if f.code == "unexpanded_xacro"]
    # Both ${...} and $(find ...) patterns should fire.
    assert len(matches) == 2
    for match in matches:
        assert match.severity == "warning"


# ---------------------------------------------------------------------------
# Diagnostics API surface
# ---------------------------------------------------------------------------


def test_diagnostics_fatal_property_filters_correctly() -> None:
    diag = inspect_urdf(_UNDECLARED_LINK_URDF)
    assert all(f.severity == "fatal" for f in diag.fatal)


def test_diagnostics_warnings_property_filters_correctly() -> None:
    diag = inspect_urdf(_NON_UNIT_AXIS_URDF)
    assert all(f.severity == "warning" for f in diag.warnings)


def test_diagnostics_messages_format() -> None:
    """messages() returns one bracketed-prefix line per finding."""
    diag = inspect_urdf(_UNDECLARED_LINK_URDF)
    msgs = diag.messages()
    assert len(msgs) == len(diag.findings)
    for msg in msgs:
        assert msg.startswith("[FATAL]") or msg.startswith("[WARNING]")


def test_diagnostics_is_fatal_false_for_warnings_only() -> None:
    diag = inspect_urdf(_NON_UNIT_AXIS_URDF)
    assert not diag.is_fatal


def test_diagnostics_is_fatal_true_for_undeclared_ref() -> None:
    diag = inspect_urdf(_UNDECLARED_LINK_URDF)
    assert diag.is_fatal


def test_diagnostics_frozen() -> None:
    """Diagnostics is immutable so callers can cache it."""
    diag = inspect_urdf(_CLEAN_URDF)
    with pytest.raises((AttributeError, TypeError)):
        diag.findings = ()  # type: ignore[misc]


def test_finding_frozen() -> None:
    """Finding is immutable."""
    finding = Finding(
        severity="warning",
        code="test_code",
        message="test",
        names=("a",),
    )
    with pytest.raises((AttributeError, TypeError)):
        finding.code = "other"  # type: ignore[misc]
