function displayme(){
var x = parseInt(document.getElementById("field1").value);
var y = parseInt(document.getElementById("field2").value);
var hammingDistance = function(x,y){
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
document.getElementById("output").value = hammingDistance(x,y);
}
