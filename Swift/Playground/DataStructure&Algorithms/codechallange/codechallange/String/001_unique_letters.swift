//
//  001_unique_letters.swift
//  codechallange
//
//  Created by Toan Hoang on 8/12/20.
//  Copyright © 2020 Toan Hoang. All rights reserved.
//

import Foundation

// Code
func challenge1a(input: String) -> Bool {
    var usedLeters = [Character]()
    
    for letter in input {
        if usedLeters.contains(letter) {
            return false
        }
        
        usedLeters.append(letter)
    }
    
    return true
}


// Test case
func main() {
    print("Test Case")
    assert(challenge1a(input: "No duplicates") == true, "Failed")
    assert(challenge1a(input: "Hello world") == false, "Failed")
}
