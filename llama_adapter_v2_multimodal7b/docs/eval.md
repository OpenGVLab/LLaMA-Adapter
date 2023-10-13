# Evaluation on MME Benchmark

[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) is a comprehensive evaluation benchmark for multimodal large language models. It measures both perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster, celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation, and code reasoning.

## Setup & Evaluation

1. Download MME datasets and `eval_tool` from the [MME repo](https://github.com/bradyfu/awesome-multimodal-large-language-models#our-mllm-works), and put them under `MME_Benchmark_release_version`. Now the folder structure will be:
    ```
    MME_Benchmark_release_version
        ├── artwork
        ├── celebrity
        ├── code_reasoning
        ├── color
        ├── commonsense_reasoning
        ├── count
        ├── eval_tool
        │   ├── calculation.py
        │   ├── LaVIN
        │   └── Your_Results
        ├── existence
        ├── landmark
        ├── numerical_calculation
        ├── OCR
        ├── position
        ├── posters
        ├── scene
        └── text_translation
    ```
2. Generate MME results using: `python util/evaluate_mme.py --pretrained_path [MODEL_PATH] --llama_path [LLAMA_DIR] --output_path [RESULT_FILE_PATH]`
3. Evaluate LLaMA-Adapter V2.1 with MME's eval_tool: `python MME_Benchmark_release_version/eval_tool/calculation.py --results_dir [RESULT_FILE_PATH]`

## Results

> For comparisons with other works, please check [MME Leaderboard](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).

* **LLaMA-Adapter V2.1**

    ```
    =========== Perception ===========
    total score: 1326.0875953396435 

            existence  score: 185.0
            count  score: 133.33333333333331
            position  score: 56.666666666666664
            color  score: 118.33333333333334
            posters  score: 147.9591836734694
            celebrity  score: 134.70588235294116
            scene  score: 156.25
            landmark  score: 167.8391959798995
            artwork  score: 123.5
            OCR  score: 102.5


    =========== Cognition ===========
    total score: 356.42857142857144 

            commonsense_reasoning  score: 106.42857142857144
            numerical_calculation  score: 47.5
            text_translation  score: 112.5
            code_reasoning  score: 90.0

    ```
