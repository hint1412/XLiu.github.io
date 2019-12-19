/*
variables
*/
var model;

/*
display after click
*/
function displayme(){
    var a = parseFloat(document.getElementById("input_a").value) || 0.0;
    var b = parseFloat(document.getElementById("input_b").value) || 0.0;
    var w = parseFloat(document.getElementById("input_w").value) || 0.0;
    var l0 = parseFloat(document.getElementById("input_l0").value) || 0.0;
    var l1 = parseFloat(document.getElementById("input_l1").value) || 0.0;
    var P = parseFloat(document.getElementById("input_P").value) || 0.0;    
    var KP = function(a,b,w,l0,l1){
        var inputs = tf.tensor([[a/b, w/b, l0/b, l1/b]]);
        var outputs = model.predict(inputs).dataSync();
        var kop = outputs*l1*Math.pow(a, 0.5)/b/w/w;
        return kop;
    };
    var KP_o = KP(a,b,w,l0,l1);
    var K_o = KP(a,b,w,l0,l1)*P;
    document.getElementById("output_KP").value = KP_o.toFixed(3);
    document.getElementById("output_K").value = K_o.toFixed(3);
}

/*
load the model
*/
async function start() { 
    //load the model
    model = await tf.loadLayersModel('models/model.json', strict=false)
    
    //warm up 
    var a = tf.tensor([[1, 1, 1, 1]]);
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
