function init_state(){
    const f16 = new Float32Array([-0.5014382004737854, 0.745929479598999, 1.004421353340149, -1.0390024185180664, -0.3761645555496216, -0.6434043645858765, 4.036914825439453, 0.7974051237106323, -3.7382349967956543, -0.26134952902793884, 1.162588357925415, 0.7374235987663269, 0.6095333695411682, 1.9149888753890991, -0.6184295415878296, -0.449447333812713]);
    const init_tensor = new ort.Tensor("float32", f16, [1, 1, 16])
    return init_tensor;
}

function init_zeros(){
    const f3_1_64 = new Float32Array(3*1*128);
    const zero_tensor = new ort.Tensor("float32", f3_1_64, [3, 1, 128]);
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

function unnormalizeTensor(t){
    const mu_diff = [-9.3696e-05, -1.0775e-04,  7.2377e-05, -2.5050e-04,  1.4969e-06,
        3.9958e-09,  6.1189e-03,  7.2579e-04, -6.3177e-03, -1.5325e-03,
        4.3263e-04, -4.7129e-04,  8.9154e-09,  2.3711e-03, -8.3655e-09,
        1.1528e-04];
    const sigma_diff = [1.6201e-01, 3.0381e-01, 6.5666e-02, 7.8177e-02, 9.0162e-04, 1.2879e-06,
        1.4171e+00, 5.0086e-01, 1.4407e+00, 4.9621e-01, 3.5076e-01, 8.5247e-01,
        2.6876e-06, 3.4706e-01, 2.3001e-06, 2.8259e-01];
    let f16 = new Float32Array(16);
    for(let i = 0; i < 16; i++){
        f16[i] = mu_diff[i] + t.cpuData[i] * sigma_diff[i];
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
    const unnormalized_output = unnormalizeTensor(results['output']);
    state = addTensors(state, unnormalized_output);
    return unnormalized_output;
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
        const t1 = performance.now();
        if(playing){
            await run_lstm_inference(lstm_session);
            for(let i = 0; i < 16; i++){
                sliders[i].value = state.cpuData[i];
                labels[i].innerHTML = state.cpuData[i].toFixed(2);
            }
        }
        const decoded_img = await run_decoder_inference(decoder_session);
        drawTensor(decoded_img);
        const elapsed = performance.now() - t1;
        document.getElementById("stats").innerHTML = `CPU Time / Frame: ${elapsed.toFixed(0)} ms, Max FPS: ${(1000/elapsed).toFixed(0)}`;
    }, 70);
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