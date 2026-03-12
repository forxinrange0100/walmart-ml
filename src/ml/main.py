from ml.data_pipeline.main import main as data_pipeline_main
from ml.train_pipeline.main import main as train_pipeline_main

def main() -> None:
    data_pipeline_main()
    train_pipeline_main()

if __name__ == "__main__":
    main()
