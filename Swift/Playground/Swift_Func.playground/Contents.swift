import UIKit

func minMax(array: [Int]) -> (min: Int, max: Int){
    var currentMin = array[0]
    var currentMax = array[0]
    
    for value in array[1..<array.count]{
        currentMin = min(currentMin, value)
        currentMax = max(currentMax, value)
    }
    
    return (currentMin, currentMax)
}

let bounds = minMax(array: [8, -6, 2, 109, 3, 72])

print("\(bounds.min)")
print("\(bounds.max)")
