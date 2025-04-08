
const canvas = document.getElementById("projectCanvas") as HTMLCanvasElement;
canvas.width = window.innerWidth*devicePixelRatio;
canvas.height = window.innerHeight*devicePixelRatio;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
if(!ctx){
    throw Error("Context unable to be found");
}



const xlen = 4 //From  0 to 4
const dr = 0.001 //Step size
const RSU_W = 0.8 //Ratio of screen width used
const RSU_H = 0.8 //Ratio of screen height used
// const yScalle = 10



const performItertations = (xstart: number, r: number, iterations: number, returnIterations: number = 1):number[] => {
    if(returnIterations > iterations){
        throw Error("returnIterations must be less than or equal to iterations");
    }
    let x = xstart;
    const xVals: number[] = new Array(returnIterations).fill(null);
    
    xVals[0] = x;
    let refPoint = 1;
    for(let i = 0; i < iterations; i++){
        x = r * x * (1 - x);
        xVals[refPoint] = x;
        refPoint = (refPoint+1) - returnIterations*Number(refPoint == returnIterations-1);

    }
    const returnList: number[] = new Array(returnIterations).fill(null);
    for(let i = 0; i < returnIterations; i++){
        returnList[i] = xVals[(refPoint + i) % returnIterations];
    }
    return returnList

}


const initializeXAxis = (rmax: number, rmin: number, dx: number):number[] => {
    const xVals = new Array(Math.floor((rmax - rmin) / dx)).fill(null);
    for(let i = 0; i < xVals.length; i++){
        xVals[i] = rmin + i * dx;
    }
    return xVals;
}






