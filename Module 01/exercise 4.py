text_data = "128000 tokens, 0.005 USD per 1K"

# Pisahkan string berdasarkan tanda koma
parts = text_data.split(",")
# parts akan berisi: ['128000 tokens', ' 0.005 USD per 1K']

# Ekstrak jumlah token (ambil bagian pertama, pisahkan spasi, ambil angka pertama)
token_part = parts[0].strip()  # "128000 tokens"
token_count = int(token_part.split()[0])

# Ekstrak harga (ambil bagian kedua, hapus spasi depan, pisahkan spasi, ambil angka pertama)
cost_part = parts[1].strip()   # "0.005 USD per 1K"
cost_value = float(cost_part.split()[0])

# --- Hasil Output ---
print(f"Extracted Token Count: {token_count} (Type: {type(token_count).__name__})")
print(f"Extracted Cost: {cost_value} (Type: {type(cost_value).__name__})")