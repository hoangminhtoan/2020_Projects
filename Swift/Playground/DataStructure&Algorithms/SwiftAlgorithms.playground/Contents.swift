import UIKit

example(of: "Creating and Linking Nodes"){
    let node1 = Node(value: 1)
    let node2 = Node(value: 2)
    let node3 = Node(value: 3)
    
    node1.next = node2
    node2.next = node3
    
    print(node1)
}

example(of: "Pushing Element into the List"){
    var list = LinkedList<Int>()
    list.push(3)
    list.push(2)
    list.push(1)
    
    print(list)
}


example(of: "Appending Element into the List"){
    var list = LinkedList<Int>()
    list.append(3)
    list.append(2)
    list.append(1)
    
    print(list)
}

example(of: "Inserting at a Particular Index"){
    var list = LinkedList<Int>()
    list.push(3)
    list.push(2)
    list.append(1)
    
    print("Before Inserting: \(list)")
    var middleNode = list.node(at: 1)!
    for i in 1...4{
        middleNode = list.insert(i*2, after: middleNode)
    }
    print("After Inserting: \(list)")
}

example(of: "Removing Head Node of Linked List"){
    var list = LinkedList<Int>()
    for i in 1...5{
        list.push(i)
    }
    
    print("Before Popping List: \(list)")
    
    list.pop()
    print("After Popping List: \(list)")
}

example(of: "Removing Last Node of Linked List"){
    var list = LinkedList<Int>()
    for i in 1...5{
        list.push(i)
    }
    
    print("Before remove The Last Node: \(list)")
    
    list.removeLast()
    print("After Removing the Last Node: \(list)")
}


example(of: "Using Collection"){
    var list = LinkedList<Int>()
    
    for i in 0...9{
        list.append(i)
    }
    
    print("List: \(list)")
    print("First Element: \(list[list.startIndex])")
    
}
