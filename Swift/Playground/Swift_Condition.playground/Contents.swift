import UIKit

var str = "Hello, playground"

// Switch condition
let chr = "a"

switch chr{
case "a":
    print("This is an a")
default:
    print("This is the fallback")
}

var n: Int = 32

print("\(n & (n-1))")

// Return values in function
// func (<param>: <type>) -> <return type>{
//...
// return <return type>
//}

var a: Int = 3
var b: Int = 5
var c: Int!

c = a
a = b
b = c

print("a=\(a) b=\(b)")


// Parent class
class Employee{
    var name:String = ""
    var job:String = ""
    
    func doWork(){
        print("Hello \(name) and I am doing my work as \(job)")
    }
}

var e = Employee()
e.name = "Toan Hoang"
e.job = "AI Engineer"
e.doWork()

//  Child class
class Manager:Employee{
    override func doWork() {
        super.doWork()
        print("After 5 years working as \(self.job) I got promoted")
    }
}


var m = Manager()
m.name = "Toan Hoang"
m.job = "AI Engineer"
m.doWork()
