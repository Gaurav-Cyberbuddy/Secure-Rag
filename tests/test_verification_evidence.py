"""
Tests for list-aware verification evidence selection.

These tests do not call the LLM or NLI model.
"""

from backend.security.verification_evidence import (
    group_evidence_blocks,
    select_verification_evidence,
    format_block_for_nli,
    build_nli_candidates,
)


SOFTWARE_EVIDENCE = """
All experiments were conducted using Python and the PyTorch deep learning framework.

The implementation was developed and executed in Visual Studio Code on a Windows-based system.

The primary software components used in this research include:

* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* NumPy
* Scikit-learn

4.2 Dataset Configuration
The CIFAR-10 dataset was automatically downloaded using Torchvision.
"""


def test_structured_list_forms_one_atomic_block():
    blocks = group_evidence_blocks(SOFTWARE_EVIDENCE)

    list_blocks = [
        block
        for block in blocks
        if "Python 3.x" in block and "Scikit-learn" in block
    ]

    assert len(list_blocks) == 1

    block = list_blocks[0]

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in block


def test_select_verification_evidence_keeps_complete_list():
    claim = (
        "The research used Python 3.x, PyTorch, Torchvision, "
        "Matplotlib, NumPy, Scikit-learn, and Visual Studio Code "
        "on a Windows-based system."
    )

    selected = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
        "Visual Studio Code",
    ):
        assert item in selected


def test_format_block_for_nli_does_not_drop_list_items():
    block = """
The primary software components used in this research include:
* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* NumPy
* Scikit-learn
"""

    formatted = format_block_for_nli(block)

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in formatted


def test_bare_list_without_bullets_is_preserved():
    evidence = """
The primary software components used in this research include:
Python 3.x
PyTorch
Torchvision
Matplotlib
NumPy
Scikit-learn
"""

    selected = select_verification_evidence(
        claim=(
            "The research used Python 3.x, PyTorch, Torchvision, "
            "Matplotlib, NumPy, and Scikit-learn."
        ),
        evidence=evidence,
    )

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in selected


def test_build_nli_candidates_puts_complete_list_first():
    claim = (
        "The research used Python 3.x, PyTorch, Torchvision, "
        "Matplotlib, NumPy, and Scikit-learn."
    )

    evidence = (
        SOFTWARE_EVIDENCE
        + "\n\n--- EVIDENCE ---\n\n"
        + "Unrelated chunk about evaluation metrics only."
    )

    candidates = build_nli_candidates(
        claim=claim,
        evidence=evidence,
    )

    assert candidates
    first = candidates[0]

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in first


def test_pdf_linewrap_preserves_visual_studio_with_list():
    evidence = """
All experiments were conducted using Python and the PyTorch deep learning framework.
The implementation was developed and executed in Visual Studio Code on a Windows-
based system. Model training, adversarial attack generation, robustness evaluation, and
result visualization were performed within a virtual Python environment.
The primary software components used in this research include:
* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* NumPy
* Scikit-learn
"""

    selected = select_verification_evidence(
        claim=(
            "The research used Python 3.x, PyTorch, Torchvision, "
            "Matplotlib, NumPy, Scikit-learn, and Visual Studio Code "
            "on a Windows-based system."
        ),
        evidence=evidence,
    )

    assert "Visual Studio Code" in selected
    assert "Windows-based" in selected

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in selected
