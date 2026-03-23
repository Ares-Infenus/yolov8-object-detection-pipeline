"""
Verify all Python scripts have valid syntax (no import errors at parse time).
"""
import pytest
import py_compile
from pathlib import Path


def get_python_files():
    files = []
    for directory in ["src", "scripts", "tests"]:
        files.extend(Path(directory).glob("*.py"))
    return files


class TestScriptsSyntax:
    @pytest.mark.parametrize("pyfile", get_python_files(), ids=lambda p: str(p))
    def test_valid_syntax(self, pyfile):
        try:
            py_compile.compile(str(pyfile), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in {pyfile}: {e}")
