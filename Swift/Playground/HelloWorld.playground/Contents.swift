import UIKit

var str = "Hello, playground"

// create list
let names = ["Anna", "Alex", "Brian", "Jack"]
let count = names.count

/*
for i in 0..<coun{
    print("Person \(i+1) is called \(names[i])\n")
}
*/

for name in names{
    print(name)
}

let precomposed: Character = "\u{D55C}"

let decomposed: Character = "\u{1112}\u{1161}\u{11AB}"
print(decomposed)

