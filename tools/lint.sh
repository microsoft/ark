#!/usr/bin/env bash
set -o pipefail

PROJECT_ROOT=$(dirname "$(realpath "$0")")/..
LINT_CPP=false
LINT_PYTHON=false
DRY_RUN=false
EXIT_CODE=0

usage() {
    echo "Usage: $0 [cpp] [py] [dry]"
    echo "  cpp     Lint C++ code"
    echo "  py      Lint Python code"
    echo "  dry     Dry run mode (no changes made)"
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        cpp)
            LINT_CPP=true
            ;;
        py)
            LINT_PYTHON=true
            ;;
        dry)
            DRY_RUN=true
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            usage
            exit 1
            ;;
    esac
done

# If no cpp or py specified, default to both
if [ "$LINT_CPP" = false ] && [ "$LINT_PYTHON" = false ]; then
    LINT_CPP=true
    LINT_PYTHON=true
fi

if $LINT_CPP; then
    echo "Linting C++ code..."
    # Find all git-tracked files with .c/.h/.cpp/.hpp/.cc/.cu/.cuh extensions
    # Exclude third_party/ to match project convention (not our code to format)
    files=$(git -C "$PROJECT_ROOT" ls-files --cached | grep -E '\.(c|h|cpp|hpp|cc|cu|cuh)$' | grep -v -E '(third_party/)' | sed "s|^|$PROJECT_ROOT/|")
    if [ -n "$files" ]; then
        if $DRY_RUN; then
            if ! echo "$files" | xargs -d '\n' clang-format -style=file --dry-run --Werror; then
                EXIT_CODE=1
            fi
        else
            if ! echo "$files" | xargs -d '\n' clang-format -style=file -i; then
                EXIT_CODE=1
            fi
        fi
    fi
fi

if $LINT_PYTHON; then
    echo "Linting Python code..."
    # Find all git-tracked files with .py extension
    # Exclude paths matching pyproject.toml [tool.black] exclude patterns;
    # black's exclude regex only applies during directory traversal, not to
    # explicitly-listed files, so we must filter them out here.
    files=$(git -C "$PROJECT_ROOT" ls-files --cached | grep -E '\.py$' | grep -v -E '(third_party/|\.eggs/|docs/|examples/llama/llama/)' | sed "s|^|$PROJECT_ROOT/|")
    if [ -n "$files" ]; then
        if $DRY_RUN; then
            if ! echo "$files" | xargs -d '\n' python3 -m black --check --diff --config "$PROJECT_ROOT/pyproject.toml"; then
                EXIT_CODE=1
            fi
        else
            if ! echo "$files" | xargs -d '\n' python3 -m black --config "$PROJECT_ROOT/pyproject.toml"; then
                EXIT_CODE=1
            fi
        fi
    fi
fi

exit $EXIT_CODE
