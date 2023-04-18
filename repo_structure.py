import os


def create_directory_structure(base_path, structure):
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        os.makedirs(path, exist_ok=True)

        if isinstance(value, dict):
            create_directory_structure(path, value)



if __name__ == "__main__":
    base_path = ""
    directory_structure = {
        "data": {"raw": {},
                 "clean":{},
                 "processed": {}},
        "models": {},
        "notebooks": {},
        "src": {
            "data": {},
            "features": {},
            "models": {},
            "evaluation": {},
            "visualization": {},
        },
        "tests": {},
    }

    create_directory_structure(base_path, directory_structure)
