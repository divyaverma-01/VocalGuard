import yaml
from utils.seed import set_seed

def main():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    print("Project initialized successfully.")

if __name__ == "__main__":
    main()
