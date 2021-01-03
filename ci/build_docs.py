from Peeves.Doc import *
import os

root = os.path.dirname(os.path.dirname(__file__))
target = os.path.join(root, "docs")
doc_config = {
    "config": {
        "title": "McUtils",
        "path": "McUtils",
        "url": "https://mccoygroup.github.io/McUtils/",
        "gh_username": "McCoyGroup",
        "footer": "Brought to you by the McCoy Group"
    },
    "packages": [
        {
            "id": "McUtils",
            'tests_root': os.path.join(root, "ci", "tests")
        }
    ],
    "root": root,
    "target": target,
    "readme": os.path.join(root, "README.md")
}
DocBuilder(**doc_config).build()