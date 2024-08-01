from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Настройка конфигурации квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Проверка наличия GPU и установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Указание контрольной точки модели
checkpoint = "IlyaGusev/saiga_llama3_8b"

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Загрузка квантизированной модели
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto"
)

# Перевод модели в режим оценки
model.eval()

# Пример текста для генерации
input_text = "Как научиться писать код?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Генерация текста
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=8,  # Максимальная длина генерируемого текста
        num_return_sequences=1,  # Количество генерируемых последовательностей
        do_sample=True,  # Включение сэмплинга для разнообразия текста
        top_k=10,  # Использование top-k sampling
        top_p=0.95,  # Использование top-p (nucleus) sampling
        temperature=0.5  # Температура сэмплинга
    )

# Декодирование и вывод генерируемого текста
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

inputs = tokenizer(input_text, return_tensors="pt").to(device)
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=5,  # Максимальная длина генерируемого текста
            num_return_sequences=1,  # Количество генерируемых последовательностей
            do_sample=True,  # Включение сэмплинга для разнообразия текста
            top_k=5,  # Использование top-k sampling
            top_p=0.95,  # Использование top-p (nucleus) sampling
            temperature=0.6  # Температура сэмплинга
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

    
response = generate_response("Как написать алгоритм быстрого поиска?")
print("Модель: ", response)