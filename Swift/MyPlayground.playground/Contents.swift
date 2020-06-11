import UIKit

var str = "Hello, playground"

class Dog{
    var name = ""
    func bark(){
        print("woof woof!")
    }
}

let dog1 = Dog()
dog1.name = "Fido"
print(dog1.name)
dog1.bark()
