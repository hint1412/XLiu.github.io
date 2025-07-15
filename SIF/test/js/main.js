/*
variables
*/
var model;
var record_num = 0;

/*
display after click
*/
function displayme() {
    // record number
    record_num += 1;
    
    // inputs
    var input_1 = parseFloat(document.getElementById("input_1").value) || 0.0;
    var input_2 = parseFloat(document.getElementById("input_2").value) || 0.0;
    var input_3 = parseFloat(document.getElementById("input_3").value) || 0.0;
    var input_4 = parseFloat(document.getElementById("input_4").value) || 0.0;
    var input_5 = parseFloat(document.getElementById("input_5").value) || 0.0;
    var input_6 = parseFloat(document.getElementById("input_6").value) || 0.0;
    var input_7 = parseFloat(document.getElementById("input_7").value) || 0.0;
    var input_8 = parseFloat(document.getElementById("input_8").value) || 0.0;
    var input_9 = parseFloat(document.getElementById("input_9").value) || 0.0;
    const elem_outk = document.getElementById("output_K");
    const elem_recd = document.getElementById("txtarea");
    
    // evaluation    
    function KP(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9) {
        var inputs = tf.tensor([[input_1, input_2/10.0, input_3/10.0, input_4*10.0, input_5*10.0, input_6/10.0, input_7/10.0, input_8/10.0, input_9/1000.0]]);
        var outputs = model.predict(inputs).dataSync();
        var kop = outputs[0];       
        return isFinite(kop) ? kop : 0.0;
    };
    var K_o = KP(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9);
    
    // log: C, Mn, Si, P, S, Cu, Ni, Cr
    var str_output = "C" + input_1.toFixed(3).toString() + 
                      ", Mn" + input_2.toFixed(3).toString() +
                      ", Si" + input_3.toFixed(3).toString() +
                      ", P" + input_4.toFixed(3).toString() +
                      ", S" + input_5.toFixed(3).toString() +
                      ", Cu" + input_6.toFixed(3).toString() +
		      ", Ni" + input_7.toFixed(3).toString() +
                      ", Cr" + input_8.toFixed(3).toString() +
                      ", T" + input_9.toFixed(3).toString() +
                      ", KI=" + K_o.toFixed(3).toString();
    console.log(str_output);    
                           
    // outputs 
    elem_outk.style.color = "black";
    elem_recd.innerHTML += "Record " + pad(record_num, 3) + 
                           ":  " + str_output + 
                           ";" + "&#13;&#10;";        
        
    elem_outk.value = K_o.toFixed(3);      
}

/*
zero padding
*/
function pad(n, width, z) {
  z = z || '0';
  n = n + '';
  return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
}

/*
load the model
*/
async function start() { 
    //load the model
    model = await tf.loadLayersModel('models/model.json')
    
    //warm up 
    var a = tf.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]);
    console.log('a shape:', a.shape, a.dtype);
    var pred = model.predict(a).dataSync();
    console.log('pred:', pred);
    
    //allow running
    allowrun()
}

/*
allow input
*/
function allowrun() {
    document.getElementById('status').innerHTML = 'Model Loaded';
    document.getElementById("run").disabled = false;
}
