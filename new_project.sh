#!/usr/bin/env bash
#
# Start a new analytics project from MSE803-Analytics-Template.
#
#   ./new_project.sh W5Act1                    -> ./W5Act1
#   ./new_project.sh W5Act1 W5                 -> ./W5/W5Act1
#   ./new_project.sh W5Act1 ~/projects         -> ~/projects/W5Act1
#   ./new_project.sh W5Act1 W5 --no-prefix     -> keeps 01_data_understanding.ipynb
#
# Copies the template, prefixes the notebooks to match the naming used elsewhere
# in this repo (W5Act1_01_data_understanding.ipynb), and substitutes the project
# name into README.md, CLAUDE.md, the report template and the notebook titles.
#
# Plain `cp -R MSE803-Analytics-Template somewhere/MyProject` works too.

set -euo pipefail

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/MSE803-Analytics-Template"

usage() {
    echo "usage: $(basename "$0") <project-name> [destination-dir] [--no-prefix]" >&2
    echo >&2
    echo "  project-name      e.g. W5Act1. Used for the folder and in the docs." >&2
    echo "  destination-dir   where to create it (default: current directory)." >&2
    echo "  --no-prefix       keep notebook names as 01_data_understanding.ipynb etc." >&2
    exit 1
}

PROJECT_NAME=""
DEST_DIR="."
PREFIX_NOTEBOOKS=1

for arg in "$@"; do
    case "$arg" in
        --no-prefix) PREFIX_NOTEBOOKS=0 ;;
        -h|--help)   usage ;;
        -*)          echo "unknown option: $arg" >&2; usage ;;
        *)
            if [ -z "$PROJECT_NAME" ]; then PROJECT_NAME="$arg"
            else DEST_DIR="$arg"; fi ;;
    esac
done

[ -n "$PROJECT_NAME" ] || usage

if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "error: template not found at $TEMPLATE_DIR" >&2
    exit 1
fi

if [[ ! "$PROJECT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: project name should be letters, digits, dot, dash or underscore." >&2
    echo "       got: $PROJECT_NAME" >&2
    exit 1
fi

TARGET="$DEST_DIR/$PROJECT_NAME"

if [ -e "$TARGET" ]; then
    echo "error: $TARGET already exists. Pick another name or remove it first." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

# Copy everything except git metadata and caches. The dotfiles (.claude, .gitignore)
# must come along, so copy the directory itself and then strip what we do not want.
cp -R "$TEMPLATE_DIR" "$TARGET"
rm -rf "$TARGET/.git" "$TARGET/.ipynb_checkpoints"
find "$TARGET" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$TARGET" -name '.DS_Store' -delete 2>/dev/null || true

# Substitute the project name everywhere the template left a placeholder.
find "$TARGET" -type f \( -name '*.md' -o -name '*.ipynb' \) -print0 \
    | xargs -0 sed -i '' "s/{{PROJECT_NAME}}/$PROJECT_NAME/g"

# Rename the report template to the project.
if [ -f "$TARGET/reports/REPORT_TEMPLATE.md" ]; then
    mv "$TARGET/reports/REPORT_TEMPLATE.md" "$TARGET/reports/$PROJECT_NAME.md"
fi

# Prefix the notebooks, matching W3Act1-2.ipynb / W4Act1.ipynb elsewhere in this repo.
if [ "$PREFIX_NOTEBOOKS" -eq 1 ]; then
    for nb in "$TARGET"/notebooks/*.ipynb; do
        [ -e "$nb" ] || continue
        mv "$nb" "$(dirname "$nb")/${PROJECT_NAME}_$(basename "$nb")"
    done
fi

if grep -rq '{{PROJECT_NAME}}' "$TARGET" 2>/dev/null; then
    echo "warning: some {{PROJECT_NAME}} placeholders were not substituted" >&2
fi

cat <<EOF

Created $TARGET

Next:

  1. Put the original data file in $TARGET/data/raw/
     Never edit it in place.

  2. Set DATASET = 'yourfile.csv' in the first cell of each notebook you use.

  3. Open the data-understanding notebook and answer DECISION 1 (the question)
     before running anything else.

  4. Work down $TARGET/CHECKLIST.md

  5. Delete the notebooks your question does not need. A descriptive project
     that never opens the modelling notebook is a finished analysis, not an
     incomplete one.

EOF
