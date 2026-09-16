def temperature_label(t: float) -> str:
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Temperature {t} is outside the allowed range of 0.0-1.0")
        
    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"

# --- Contoh Penggunaan ---
print(f"Temp 0.2 is {temperature_label(0.2)}") # precise
print(f"Temp 0.5 is {temperature_label(0.5)}") # balanced
print(f"Temp 0.9 is {temperature_label(0.9)}") # creative
# print(temperature_label(1.5)) # Ini akan memicu ValueError