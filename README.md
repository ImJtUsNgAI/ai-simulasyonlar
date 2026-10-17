# AI simülasyonlar

Sıfırdan öğrenen küçük yapay zeka simülasyonları. Hepsi `pygame` ve `numpy` ile çalışıyor.

- `football_ai.py`: pekiştirmeli öğrenmeyle futbol oynayan ajanlar
- `rl_football_sim.py`: aynı fikrin daha gelişmiş sürümü (5 seviye, ajan ekle/çıkar, hız ayarı). `config.py` ile birlikte çalışır.
- `ai_boxing.py`: iki karakter sıfırdan boks öğreniyor. Her uzuv ayrı hasar alıyor, yorgunluk, blok ve nakavt var.
- `fish_evolution.py`: balık sürüsü ve köpekbalığı, evrimle davranış değişimi
- `ai_football_3d.html`: tarayıcıda 3D futbol sahnesi

```
pip install pygame numpy
python ai_boxing.py
```