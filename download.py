import modal

vol = modal.Volume.from_name("qwop", create_if_missing=True)

FILE = "model_lstm_16_interp.torch"

chunk_n = 1

data = []
for chunk in vol.read_file(FILE):
    data.append(chunk)
    if chunk_n % 100 == 0:
        print(f"downloaded chunk {chunk_n}")
    chunk_n += 1

data = b''.join(data)

with open(FILE, "wb") as f:
    f.write(data)