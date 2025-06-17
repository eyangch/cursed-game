import asyncio
import base64
from PIL import Image
import numpy as np
import io
import time
import random

async def get_game_image(page):
    canvas_value = (await page.evaluate("""() => {
        return document.getElementById(\"window1\").toDataURL();
    }"""))[22:]
    return Image.open(io.BytesIO(base64.b64decode(canvas_value)))

async def append_numpy_image(page, arr, idx):
    img = (await get_game_image(page)).convert("RGB")
    arr[idx] = np.array(img)

async def append_key(content, keyboard, arr, idx, delay, active_keys):
    _, arr[idx], ret = await send_random_key(content, keyboard, delay, active_keys)
    return ret

async def process_qwop_js(route, _):
    response = await route.fetch()
    original_body = await response.text()

    modified_body = original_body[12:-4]
    modified_body = modified_body.replace("antialias:t.antialiasing>0", "antialias: t.antialiasing > 0,preserveDrawingBuffer: true")

    await route.fulfill(
        status=200,
        content_type="application/javascript",
        body=modified_body
    )

async def block_most(route, request):
    if "https://www.foddy.net" in request.url:
        await route.continue_()
    else:
        await route.abort()

async def send_random_key(gameContent, keyboard, delay, active_keys):
    key_map = 'qwop'
    tasks = []
    for i in range(4):
        if active_keys[i] == 1:
            if random.random() < 0.22:
                active_keys[i] = 0
                tasks.append(asyncio.create_task(keyboard.up(key_map[i])))
        else:
            if random.random() < 0.1:
                active_keys[i] = 1
                tasks.append(asyncio.create_task(keyboard.down(key_map[i])))
    await asyncio.sleep(delay/1000)
    key = []
    for i in range(4):
        if active_keys[i] == 1:
            key.append(key_map[i])
    key = ''.join(key)
    task = asyncio.gather(*tasks)
    return key, active_keys[:], task

async def check_restart(page, gameContent, delay):
    await asyncio.sleep(delay/1000)
    has_fallen = await page.evaluate("() => m.core.game.fallen")
    if has_fallen:
        await page.evaluate("() => {m.core.game.highScore = 0;}")
        await gameContent.press('r')        

def get_numpy_keys(keys):
    key_dict = {
        'q': 0,
        'w': 1,
        'o': 2,
        'p': 3,
        'z': 4
    }
    one_hot_keys = [[0 for _ in range(5)] for _ in keys]
    for i, key in enumerate(keys):
        one_hot_keys[i][key_dict[key]] = 1
    key_arr = np.array(one_hot_keys)
    return key_arr

async def setup_page(context):
    page = await context.new_page()
    await page.route('**/*', block_most)
    await page.route("https://www.foddy.net/QWOP.min.js", process_qwop_js)
    await page.goto("https://www.foddy.net/Athletics.html")
    await page.wait_for_selector("#window1")
    await asyncio.sleep(5)
    gameContent = page.locator("#window1")
    print(gameContent)
    await gameContent.click()
    await asyncio.sleep(0.5)
    await page.evaluate("""() => {
        document.getElementById("window1").width=96;
        document.getElementById("window1").height=96;
        C.modules.opengl.web.GL.current_context.viewport(0, 0, 96, 96);
        m.core.frame_time /= 4;
    }""")
    return page

async def play_actions(page, num, stop_signal=lambda: False):
    ret, keys = [None for _ in range(num)], [None for _ in range(num)]
    active_keys = [0, 0, 0, 0]

    gameContent = page.locator("#window1")
    await gameContent.press('r')

    last_reset = 0

    t0 = time.time()
    ptask = None
    for i in range(num):
        DELAY = 25

        chk_restart = asyncio.create_task(check_restart(page, gameContent, 0))
        get_img = asyncio.create_task(append_numpy_image(page, ret, i))
        task = await asyncio.create_task(append_key(gameContent, page.keyboard, keys, i, DELAY, active_keys))

        if ptask != None:
            await ptask
        ptask = task
        await chk_restart
        await get_img

        if i > 0:
            img_diff = ret[i] - ret[i-1]
            if np.mean(img_diff) > 40 and i - last_reset > 5:
                await check_restart(page, gameContent, 10)
                last_reset = i

        if i == num-1:
            print(f"got frame {i} with key {keys[i]} (it/sec={(i+1)/(time.time()-t0)})", end="\n")

        if stop_signal():
            break
    await ptask

    for key in 'qwop':
        await page.keyboard.up(key)
        print(f"up {key}", end=" ")
    print()

    ret = np.array(ret)
    print(f"avg pixel diff: {np.mean(ret[1:]-ret[:-1])}")

    return ret, np.array(keys)