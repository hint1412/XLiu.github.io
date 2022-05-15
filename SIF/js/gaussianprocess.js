/*
Follow the general framework from 'https://github.com/scotthellman/gaussianprocess_js'
*/
/*
variables
*/
var GPR;
var record_num = 0;

var beta = 0.3
var sigma0 = 0.0001
var sigmaf = 1.0
var sigmal = 1.0

var vx = [
	$V([0.000000000000000,0.000000000000000]),
	$V([0.000000000000000,0.266666666666667]),
	$V([0.000000000000000,0.500000000000000]),
	$V([0.000000000000000,0.766666666666667]),
	$V([0.000000000000000,1.000000000000000]),
	$V([0.255555555555556,0.000000000000000]),
	$V([0.255555555555556,0.266666666666667]),
	$V([0.255555555555556,0.500000000000000]),
	$V([0.255555555555556,0.766666666666667]),
	$V([0.255555555555556,1.000000000000000]),
	$V([0.500000000000000,0.000000000000000]),
	$V([0.500000000000000,0.266666666666667]),
	$V([0.500000000000000,0.500000000000000]),
	$V([0.500000000000000,0.766666666666667]),
	$V([0.500000000000000,1.000000000000000]),
	$V([0.755555555555556,0.000000000000000]),
	$V([0.755555555555556,0.266666666666667]),
	$V([0.755555555555556,0.500000000000000]),
	$V([0.755555555555556,0.766666666666667]),
	$V([0.755555555555556,1.000000000000000]),
	$V([1.000000000000000,0.000000000000000]),
	$V([1.000000000000000,0.266666666666667]),
	$V([1.000000000000000,0.500000000000000]),
	$V([1.000000000000000,0.766666666666667]),
	$V([1.000000000000000,1.000000000000000])
];

var vy = $M([
	[0.1563],
	[0.1499],
	[0.1446],
	[0.1400],
	[0.1466],
	[0.2769],
	[0.2897],
	[0.3054],
	[0.3207],
	[0.3428],
	[0.3722],
	[0.3851],
	[0.3973],
	[0.4195],
	[0.4345],
	[0.4203],
	[0.4249],
	[0.4312],
	[0.4451],
	[0.4539],
	[0.4456],
	[0.4476],
	[0.4475],
	[0.4516],
	[0.4593]
]);  

/*
Kernerls
*/
var Kernels = function(){	
    // Basic kernel functions: constant
	function constant(x,y){
		return 1;
	}
    // Basic kernel functions: linear
	function linear(x,y){
		return x.dot(y);
	}
    // Basic kernel functions: squaredExponential
	function squaredExponential(x,y,l){
		var diff = x.subtract(y);
		diff = diff.dot(diff);
		return Math.exp(-0.5 * (diff/(l*l)));
	}
    // Build kernels and combine them
	function kernelBuilder(){
		var functions = arguments;
		return{
			// sum all kernel functions
			kernel: function(x,y){
				var result = 0;
				for(var i = 0; i < functions.length; i++){
					result += functions[i].kernel(x,y);
				}
				return result;
			},

			functions: functions
		}
	}

    // return
	return {
		// constant
		constant: function(theta) { 
			var parameters = [theta];
			return {
				kernel: function(x,y){return parameters[0] * constant(x,y);},
				parameters : parameters
			}
		},
        // linear
		linear: function(theta) {
			var parameters = [theta];
			return{
				kernel: function(x,y){return parameters[0] * linear(x,y);},
				parameters : parameters
			}
		},
        // squaredExponential
		squaredExponential: function(theta,l) {
			var parameters = [theta,l];
			return{
				kernel : function(x,y){return parameters[0] * parameters[0] * squaredExponential(x,y,parameters[1]);},
				parameters : parameters
			}
		},
        // kernelBuilder
		kernelBuilder: kernelBuilder 
	}
}();

/*
Gaussian Process
*/
function GaussianProcess(kernel, beta, sigma0){
	// predict
	function evaluate(training_data,training_labels,testing_data){
		// data --> covariance matrix
		var C = applyKernel(training_data,training_data,kernel);
		var In = Matrix.I(training_data.length);		
		var k = applyKernel(training_data,testing_data,kernel);
		var Cadd = C.add(In.multiply(sigma0 * sigma0)); 
		var Cinv = Cadd.inv(); 
		var c = applyKernel(testing_data,testing_data,kernel);

		// condition
		var mu = k.transpose() .x (Cinv .x (mixedadd(training_labels.elements,-beta)));
		var mu = mixedadd(mu.elements, beta);
		var sigma2 = c.subtract(k.transpose() .x (Cinv .x (k)));
		var sigma2 = mixedadd(sigma2.elements, sigma0*sigma0);

        // return
		return{
			mu:mu,
			sigma2:sigma2
		}
	}
	
    // covariance matrix
	function applyKernel(X,Y,kernel){
        // loop		
		var result_array = []
		for(var i = 0; i < X.length; i++){
			result_array.push([]);
			for(var j = 0; j < Y.length; j++){
				result_array[i].push(kernel.kernel(X[i],Y[j]));
			}
		}
        // return		
		return $M(result_array);
	}	

    // matrix/vector + scalar
	function mixedadd(X,ys){
        // loop		
		var result_array = []
		for(var i = 0; i < X.length; i++){
			result_array.push([]);
			for(var j = 0; j < X[0].length; j++){
				result_array[i].push(X[i][j] + ys);
			}
		}
        // return		
		return $M(result_array);
	}


    // return
	return{
		evaluate:evaluate,
		kernel:kernel
	}
}

/*
load the model
*/
async function start() { 
    // load the model
    var K = Kernels.kernelBuilder(Kernels.squaredExponential(sigmaf, sigmal));
    GPR = GaussianProcess(K, beta, sigma0); 	  
    
    // warm up 
    console.log('Warm up!');
    console.log('Sylvester Precision:', Sylvester.precision);

    var x1 = [$V([0.5,0.5])]

    var result = GPR.evaluate(vx,vy,x1);
    var mu = result.mu.elements[0][0];
    var sigma2 = result.sigma2.elements[0][0];
    var sigma = Math.sqrt(Math.max(sigma2,0.0));

    console.log(mu, sigma);

    //allow running
    allowrun()
}

/*
allow input
*/
function allowrun() {
	window.onload = function what() {
		document.getElementById('status').innerHTML = 'Model Loaded';
		document.getElementById("run").disabled = false;
	}
}

/*
display after click
*/
function displayme() {
    // record number
    record_num += 1;
    
    // inputs
    var E = parseFloat(document.getElementById("input_E").value) || 0.0;
    var NU = parseFloat(document.getElementById("input_NU").value) || 0.0;
    var Y = parseFloat(document.getElementById("input_Y").value) || 0.0;
    var R = parseFloat(document.getElementById("input_R").value) || 0.0;
    var P = parseFloat(document.getElementById("input_P").value) || 0.0;
    const elem_intp = document.getElementById("intp");
    const elem_outk = document.getElementById("output_K");
    const elem_outE2Y = document.getElementById("output_E2Y");
    const elem_recd = document.getElementById("txtarea");
    
    // evaluation 
    var E2Y = E/Y   
    var x1 = [$V([(E2Y-10.0)/90.0, (NU-0.15)/0.30])]
    var result = GPR.evaluate(vx,vy,x1);
    var mu = result.mu.elements[0][0];
    var sigma2 = result.sigma2.elements[0][0];
    var sigma = Math.sqrt(Math.max(sigma2,0.0));
    var K_o = mu/Math.pow(R,1.5)*P;
    
    
    // log
    var str_output = "E=" + E.toFixed(3).toString() + 
                      ", NU=" + NU.toFixed(3).toString() +
                      ", Y=" + Y.toFixed(3).toString() +
                      ", R=" + R.toFixed(3).toString() +
                      ", E/Y=" + E2Y.toFixed(3).toString() +
                      ", P=" + P.toFixed(3).toString() +
                      ", KI=" + K_o.toFixed(3).toString();
    console.log(str_output);   
                           
    // outputs 
    if (NU >= 0.15 && NU <= 0.45 && E2Y >= 10.0 && E2Y <= 100.0) {
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

    elem_outE2Y.value = E2Y.toFixed(3);     
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
