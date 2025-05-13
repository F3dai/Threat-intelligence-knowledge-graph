# Model Evaluation Framework

This directory contains scripts for evaluating and comparing different models for threat intelligence extraction. The framework measures performance metrics like speed, extraction quality, and cost-effectiveness.

## Evaluation Script

The `model_evaluation.py` script evaluates different models based on:
1. Speed (time to process)
2. Entity extraction quality (number of nodes/relationships)
3. Cost efficiency (cost per entity extracted)

### Usage

```bash
# Evaluate all models on default articles
python3 evaluation/model_evaluation.py

# Evaluate specific models
python3 evaluation/model_evaluation.py --models gpt-4o claude-3-5-sonnet-20240620

# Use specific articles 
python3 evaluation/model_evaluation.py --articles https://your-article-url.com/article1 https://your-article-url.com/article2

# Run evaluations in parallel (faster but uses more API tokens)
python3 evaluation/model_evaluation.py --parallel

# Enable verbose logging
python3 evaluation/model_evaluation.py --verbose

# Specify custom output directory (instead of timestamped folder)
python3 evaluation/model_evaluation.py --output-dir my_custom_eval_folder
```

## Interactive Visualizations

The evaluation script now automatically generates interactive visualizations and a comprehensive HTML report. The generated visualizations include:

1. Interactive bar charts comparing processing time
2. Interactive bar charts comparing entity extraction
3. Interactive cost efficiency analysis
4. Interactive cost vs. entities scatter plot
5. Comprehensive HTML dashboard with all metrics and visualizations

## Output Organization

Evaluation results are organized in timestamped directories:
```
evaluation/
├── results/
│   ├── eval_20250510_123456/
│   │   ├── evaluation_results.json  # Raw evaluation data
│   │   └── charts/                  # Visualization outputs
│   │       ├── processing_time.html
│   │       ├── entity_extraction.html
│   │       ├── cost_efficiency.html
│   │       ├── cost_vs_entities.html
│   │       ├── summary_table.html
│   │       └── evaluation_report.html  # Full interactive dashboard
│   └── eval_20250510_234567/
│       └── ...
└── ...
```

## Supported Models

The framework supports evaluating:
- OpenAI models: gpt-3.5-turbo, gpt-4-turbo, gpt-4o
- Google models: 
  - Gemini 1.5: gemini-1.5-pro-latest, gemini-1.5-flash-latest
  - Gemini 2.0: gemini-2.0-flash
  - Gemini 2.5: gemini-2.5-pro-preview-03-25, gemini-2.5-flash-preview-04-17
- Anthropic models: claude-3-5-haiku-latest, claude-3-5-sonnet-20240620
- Baseline rule-based extractor: ner

## Example Workflow

1. Run an evaluation of all models:
   ```bash
   python3 evaluation/model_evaluation.py
   ```

2. Open the interactive dashboard in your browser:
   ```bash
   open evaluation/results/eval_YYYYMMDD_HHMMSS/charts/evaluation_report.html
   ```

## Features

- **Automatic Visualization**: Visualizations are automatically generated as part of the evaluation
- **Interactive Charts**: All charts are interactive with hover information, zoom capabilities, and filtering
- **Comprehensive Dashboard**: A single HTML dashboard includes all metrics and visualizations
- **Highlighted Best Performers**: Key metrics are highlighted to quickly identify the best models
- **In-depth Analysis**: Detailed breakdowns of performance across multiple dimensions

## Extending the Framework

To evaluate new models:
1. Add the model to the `MODELS` list in `model_evaluation.py`
2. Add pricing information to `MODEL_COSTS` in `model_evaluation.py` 
3. Ensure the model is properly handled in the `initialize_extractor` function

## Dependencies

The visualization component requires:
- plotly
- pandas
- numpy

These dependencies are included in the project's requirements.txt file.