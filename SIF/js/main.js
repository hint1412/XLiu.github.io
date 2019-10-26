/*
variables
*/
var model;

/*
display after click
*/
function displayme(){
    var x = parseInt(document.getElementById("field1").value);
    var y = parseInt(document.getElementById("field2").value);
    var sif = function(x,y){
        var count = 0;
        var n = x ^ y ;
        while ( n> 0 ){
            if ((n&1)==1){
                count += 1;
            }
            n>>=1;
        }
        return count;
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
    const a = tf.tensor([[1, 2]]);
    console.log('a shape:', a.shape, a.dtype);
    const pred = model.predict(a).dataSync();
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

