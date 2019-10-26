/*
variables
*/
var model;

/*
display after click
*/
function displayme(){
    var a = parseFloat(document.getElementById("field1").value);
    var b = parseFloat(document.getElementById("field2").value);
    var w = parseFloat(document.getElementById("field3").value);
    var sif = function(a,b,w){
        var inputs = tf.tensor([[a/b, w/b]]);
        var outputs = model.predict(inputs).dataSync();
        return outputs;
    };
    document.getElementById("output").value = sif(a,b,w);
}

/*
load the model
*/
async function start() { 
    //load the model
    model = await tf.loadLayersModel('models/model.json')
    
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

