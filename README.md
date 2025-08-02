# Finetune Qwen Chat Models

This repository contains a simplified training pipeline for fine‑tuning
[Qwen](https://huggingface.co/Qwen) chat models using the
transformers/peft stack.  The original codebase has been streamlined to
remove large data files and unused scripts while adding a few
conveniences such as on‑the‑fly dataset generation from web pages and a
minimal OpenAI‑compatible API server.

## Key Features

- **Flexible Data Input** – provide your own JSONL training/validation
  files via `--train_file` and `--eval_file` or supply a list of URLs
  using `--urls`/`--url_file` and the script will download and build a
  chat dataset automatically.
- **LoRA Fine‑Tuning** – enable parameter efficient training with the
  `--use_lora` flag.  Adapter weights are saved separately so that they
  can be applied on top of the base model during inference.
- **OpenAI‑Compatible Inference** – after training, spin up the
  FastAPI server included in the Docker compose setup and connect it to
  [Open WebUI](https://github.com/open-webui/open-webui) via
  `http://localhost:8000/v1`.

## Quick Start

1.  Install requirements

    ```sh
    pip install -r requirements.txt
    ```

2.  Fetch some training data from the web and fine‑tune the model

    ```sh
    python train.py \
      --base_model_name Qwen/Qwen1.5-7B \
      --urls https://en.wikipedia.org/wiki/Artificial_intelligence,https://en.wikipedia.org/wiki/Machine_learning \
      --output_dir model_output \
      --use_lora \
      --num_epochs 1
    ```

3.  Запустіть усі сервіси через Docker Compose

    ```sh
    docker compose up
    ```

4.  Відкрийте [Open WebUI](http://localhost:8080), перейдіть до Admin → Connections,
    створіть новий зв'язок та у полі URL вкажіть `http://localhost:8000/v1`.
    Після збереження оберіть модель у меню та почніть спілкування.

## Підтримка офлайн/онлайн режимів

Скрипт `train.py` може працювати як з підключенням до Інтернету, так і без нього:

- **Онлайн режим (за замовчуванням):** Ви можете передавати `--urls` або `--url_file`, і
  скрипт завантажить вміст сторінок за цими адресами. Базова модель буде
  автоматично завантажена з Hugging Face Hub за назвою в параметрі
  `--base_model_name`.
- **Офлайн режим:** Запустіть `train.py` із прапорцем `--offline` і вкажіть
  шлях до локальної копії моделі за допомогою `--model_path`. У цьому
  режимі жодні мережеві запити не виконуються, а для навчання будуть
  використані лише файли у `data/custom/texts`. Якщо у `data/custom/urls.txt`
  вказані якісь адреси, вони будуть проігноровані. Скрипт також автоматично
  встановлює змінні оточення `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` та
  `TRANSFORMERS_OFFLINE=1`, щоб повністю вимкнути звернення до Hugging Face Hub.

Щоб підготувати локальну модель для офлайн‑режиму, завантажте її
заздалегідь (наприклад, на іншому комп’ютері) за допомогою
`huggingface-cli snapshot-download <model-name>` або іншого інструмента
та скопіюйте отриману папку у проєкт. Потім передайте шлях до неї в
`--model_path`.

## Docker Deployment

Файл `docker-compose.yml` створює три сервіси: `finetune` для навчання,
`api` для обслуговування моделі та `openwebui` для веб‑інтерфейсу.
Після запуску `docker compose up` модель буде натренована, API‑сервер
автоматично запуститься на `http://localhost:8000/v1`, а Open WebUI буде
доступним за адресою `http://localhost:8080`.

## Citation

The inference server is compatible with any client that speaks the
OpenAI Chat API as described in the official Open WebUI docs
【362441291072753†L54-L131】.  Training code uses the `Trainer` class from
HuggingFace Transformers and applies the chat template provided by the
tokenizer to assemble multi‑turn conversations【523432065786204†L294-L311】.# Test change