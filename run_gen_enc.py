import asyncio
import models
import data

ae = models.load_ae("model_ae_16_interp.torch")

asyncio.run(data.gen_enc(ae))
