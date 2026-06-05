from ultralytics import YOLO
from pathlib import Path
import torch

# 1. HAYAT KURTARAN TAVSİYE: yolo11m (Medium) laptopları çok zorlar. 
# Eğer VRAM'in 6GB veya altındaysa bunu kesinlikle "yolo11s.pt" (Small) veya "yolo11n.pt" (Nano) yap!
model = YOLO("yolo11s.pt")

yaml_path = "/home/emre/Desktop/NATO/code/yaml-files/ir-model.yaml"

assert Path(yaml_path).exists(), "File doesn't exist"
full_path = Path(yaml_path).resolve()

if torch.cuda.is_available():
    try:
        model.train(
            data=str(full_path), 
            epochs=150,           
            imgsz=640,           
            batch=-1,             #VRAM dolup taşmasın diye 4'e çektik (veya -1 yapıp AutoBatch dene).
            workers=2,           
            device=0,            
            patience=50,         
            save=True,
            cache=False,         
            optimizer='AdamW',   
            lr0=0.01,            
            cos_lr=True,         
            mosaic=1.0,          # Dronelar için kalsın.
            mixup=0.0,           # LAPTOP KORUMASI: Mixup ekstra hesaplama yükü bindirir, laptopta kapalı (0.0) kalması daha serin tutar.
            amp=True             # BU KESİNLİKLE KALSIN: Hem hızı artırır hem GPU'yu rahatlatır.
        )
        
        # Eğitimin başarılı olursa test etmesi için içeri aldık
        results = model.val()  
        print(results.metrics)  
        
    except RuntimeError as e:
        print(f"Bir şeyler ters gitti baboli :/ Hata şu -> {e}")
        
else:
    # Cuda yoksa CPU ile eğitmeye kalkar, asıl o zaman laptop yanar!
    print("CUDA bulunamadı! Eğer CPU ile eğitime başlarsan laptop erir, aman diyeyim.")