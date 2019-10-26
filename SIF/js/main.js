/*
variables
*/
var model;

/*
display after click
*/
function displayme(){
    var x = parseFloat(document.getElementById("field1").value);
    var y = parseFloat(document.getElementById("field2").value);
    var sif = function(x,y){
        var inputs = tf.tensor([[x, y]]);
        var outputs = model.predict(inputs).dataSync();
        return outputs;
    };
    document.getElementById("output").value = sif(x,y);
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
    
    //load the class names
    //await loadDict()
}

/*
allow input
*/
function allowinput() {
    document.getElementById('status').innerHTML = 'Model Loaded';
    document.getElementById("button").disabled = false;
    //$('button').prop('disabled', false);
}

