/*
variables
*/
var model;

/*
display after click
*/
function displayme(){
    var a = parseFloat(document.getElementById("input_a").value);
    var b = parseFloat(document.getElementById("input_b").value);
    var w = parseFloat(document.getElementById("input_w").value);
    var l0 = parseFloat(document.getElementById("input_l0").value);
    var l1 = parseFloat(document.getElementById("input_l1").value);
    var P = parseFloat(document.getElementById("input_P").value);
    console.log(a,b,w,l0,l1,P);
    var KP = function(a,b,w,l0,l1){
        var inputs = tf.tensor([[a/b, w/b]]);
        var outputs = model.predict(inputs).dataSync();
        return outputs;
    };
    document.getElementById("output_KP").value = KP(a,b,w,l0,l1);
    document.getElementById("output_K").value = 0.0;
}

/*
load the model
*/
async function start() { 
    //load the model
    var model = await tf.loadLayersModel('models/model.json')
    
    //warm up 
    var a = tf.tensor([[1, 2]]);
    console.log('a shape:', a.shape, a.dtype);
    var pred = model.predict(a).dataSync();
    console.log('pred:', pred);
    
    //allow drawing on the canvas 
    allowinput()
}

/*
allow input
*/
function allowinput() {
    document.getElementById('status').innerHTML = 'Model Loaded';
    document.getElementById("button").disabled = false;
    //$('button').prop('disabled', false);
}
