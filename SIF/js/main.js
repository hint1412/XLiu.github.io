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
    var a = parseFloat(document.getElementById("input_a").value) || 0.0;
    var b = parseFloat(document.getElementById("input_b").value) || 0.0;
    var w = parseFloat(document.getElementById("input_w").value) || 0.0;
    var l0 = parseFloat(document.getElementById("input_l0").value) || 0.0;
    var l1 = parseFloat(document.getElementById("input_l1").value) || 0.0;
    var P = parseFloat(document.getElementById("input_P").value) || 0.0;
    const elem_intp = document.getElementById("intp");
    const elem_outk = document.getElementById("output_K");
    const elem_recd = document.getElementById("txtarea");
    
    // evaluation    
    var KP = function(a,b,w,l0,l1) {
        var inputs = tf.tensor([[a/b, w/b, l0/b, l1/b]]);
        var outputs = model.predict(inputs).dataSync();
        var kop = (outputs*l1*Math.pow(a, 0.5)/b/w/w);       
        return isFinite(kop) ? kop : 0.0;
    };
    var K_o = KP(a,b,w,l0,l1)*P;
    
    // log
    var str_output = "a=" + a.toFixed(3).toString() + 
                      ", b=" + b.toFixed(3).toString() +
                      ", w=" + w.toFixed(3).toString() +
                      ", L0=" + l0.toFixed(3).toString() +
                      ", L1=" + l1.toFixed(3).toString() +
                      ", P=" + P.toFixed(3).toString() +
                      ", KI=" + K_o.toFixed(3).toString();
    console.log(str_output);     
                           
    // outputs 
    if (a/b >= 0.1 && a/b <= 0.8 && w/b >= 1.0 && w/b <= 3.0 &&
        l0/b >= 0.1 && l0/b <= 0.4 && l1/b >= 2.0 && l1/b <= 5.0) {
        elem_intp.innerHTML = '<i class="far fa-grin-wink"></i> \
                               &nbsp; Interpolation';
        elem_intp.classList = "btn btn-outline-success \
                               btn-lg btn-block mb-3"; 
        elem_outk.style.color = "black";
        elem_recd.innerHTML += "Record " + pad(record_num, 3) + 
                               ":  " + str_output + " (Interpolation)" +
                               ";" + "&#13;&#10;";        
    } else {
        elem_intp.innerHTML = '<i class="far fa-frown"></i> \
                               &nbsp; Extrapolation';
        elem_intp.classList = "btn btn-outline-danger \
                               btn-lg btn-block mb-3";
        elem_outk.style.color = "#d9534f";
        elem_recd.innerHTML += "Record " + pad(record_num, 3) + 
                               ":  " + str_output + " (Extrapolation)" +
                               ";" + "&#13;&#10;";          
    }
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
    model = await tf.loadLayersModel('models/NN_4_64_64_1/model.json')
    
    //warm up 
    var a = tf.tensor([[1, 1, 1, 1]]);
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