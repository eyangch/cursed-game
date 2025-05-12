function init_state(){
    const f16 = new Float32Array([-1.5200,  7.0245,  1.0502, -2.8188,  0.4636,  1.4388, -2.5831,  1.4001,
        0.2912,  2.2350, -8.3342, -0.3677,  1.2989,  3.3506,  1.5734, -3.3443]);
    const init_tensor = new ort.Tensor("float32", f16, [1, 1, 16])
    return init_tensor;
}

function init_zeros(){
    const f3_1_64 = new Float32Array(3*1*64);
    const zero_tensor = new ort.Tensor("float32", f3_1_64, [3, 1, 64]);
    return zero_tensor;
}

let state = init_state();
let hn = init_zeros();
let cn = init_zeros();

function reset(){
    state = init_state();
    hn = init_zeros();
    cn = init_zeros();
}

let keys = [0, 0, 0, 0];
const keyDict = {
    'q': 0,
    'w': 1,
    'o': 2,
    'p': 3
};

function processKeyDown(event){
    if(event.key == 'r'){
        reset();
    }
    if(event.key in keyDict){
        keys[keyDict[event.key]] = 1;
        console.log(keys);
    }
}
function processKeyUp(event){
    if(event.key in keyDict){
        keys[keyDict[event.key]] = 0;
        console.log(keys);
    }
}

document.body.addEventListener("keydown", processKeyDown);
document.body.addEventListener("keyup", processKeyUp);

function append_key_to_state(state){
    let f20 = new Float32Array(20);
    for(let i = 0; i < 16; i++){
        f20[i] = state.cpuData[i];
    }
    for(let i = 0; i < 4; i++){
        f20[16+i] = keys[i];
    }
    return new ort.Tensor("float32", f20, [1, 1, 20]);
}

function remove_key_from_state(state){
    let f16 = new Float32Array(16);
    for(let i = 0; i < 16; i++){
        f16[i] = state.cpuData[i];
    }
    return new ort.Tensor("float32", f16, [1, 16]);
}

async function run_decoder_inference(session){
    const inputs = {
        'input': remove_key_from_state(state)
    };
    const results = await session.run(inputs);
    return results['output'];
}

function addTensors(t1, t2){
    let f16 = new Float32Array(16);
    for(let i = 0; i < 16; i++){
        f16[i] = t1.cpuData[i] + t2.cpuData[i];
    }
    return new ort.Tensor("float32", f16, [1, 1, 16]);
}

async function run_lstm_inference(session){
    const inputs = {
        'input': append_key_to_state(state),
        'h0': hn,
        'c0': cn
    };
    const results = await session.run(inputs);
    hn = results['hn'];
    cn = results['cn'];
    state = addTensors(state, results['output']);
    return results['output'];
}

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function drawTensor(data){
    const imageData = data.toImageData();
    ctx.putImageData(imageData, 0, 0);
}

let playing = true;

const slider_div = document.getElementById("sliders");
let sliders = [];
let labels = [];
for(let i = 0; i < 16; i++){
    const group_div = document.createElement("div");

    const slider = document.createElement("input");
    slider.type = "range";
    slider.style = "width: 50%";
    slider.min = -10;
    slider.max = 10;
    slider.step = "any";
    slider.value = 0;
    group_div.appendChild(slider);

    const label = document.createElement("span");
    label.innerHTML = "0.00";
    group_div.appendChild(label);

    slider.addEventListener("input", () => {
        label.innerHTML = Number(slider.value).toFixed(2);
        state.cpuData[i] = slider.value;
    });

    slider_div.appendChild(group_div);
    sliders.push(slider);
    labels.push(label);
}

async function main(){
    const lstm_session = await ort.InferenceSession.create(
        'onnx_lstm.onnx', 
        { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }
    );

    const decoder_session = await ort.InferenceSession.create(
        'onnx_decoder.onnx', 
        { executionProviders: ['cpu'], graphOptimizationLevel: 'all' }
    );

    setInterval(async () => {
        if(playing){
            await run_lstm_inference(lstm_session);
            for(let i = 0; i < 16; i++){
                sliders[i].value = state.cpuData[i];
                labels[i].innerHTML = state.cpuData[i].toFixed(2);
            }
        }
        const decoded_img = await run_decoder_inference(decoder_session);
        drawTensor(decoded_img);
    }, 75);
}

main();

for(const c of ['q', 'w', 'o', 'p', 'r']){
    const btn = document.getElementById(c);
    btn.addEventListener("mousedown", () => {
        processKeyDown({"key": c});
    });
    btn.addEventListener("mouseup", () => {
        processKeyUp({"key": c});
    });
    btn.addEventListener("touchstart", () => {
        processKeyDown({"key": c});
    });
    btn.addEventListener("touchend", () => {
        processKeyUp({"key": c});
    });
}

const play_pause = document.getElementById("play-pause");

play_pause.addEventListener("click", () => {
    if(playing){
        play_pause.innerHTML = "play";
    }else{
        play_pause.innerHTML = "pause";
    }
    playing = !playing;
});