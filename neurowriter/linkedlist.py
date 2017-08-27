#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of a doubly linked list.

@author: Álvaro Barbero Jiménez
"""


class LinkedList:
    """Implementation of a doubly linked list"""
    def __init__(self, iterable=None):
        """Initializes the linked list using an iterable"""
        iterator = iter(iterable)
        self.head = prev = LinkedListNode(next(iterator))
        for value in iterator:
            prev = LinkedListNode(value, prev)
        self.tail = prev

    def iternodes(self):
        """Iterates along the nodes in the linked list, from head to tail"""
        current = self.head
        while current is not None:
            yield current
            current = current.nxt

    def __iter__(self):
        """Iterates along the values in the linked list, from head to tail"""
        for node in self.iternodes():
            yield node.value

    def __str__(self):
        return str([x for x in self])

    def __repr__(self):
        return self.__str__()


class LinkedListNode:
    """Node of a Linked List"""

    def __init__(self, value, prev=None):
        """Creates a new Linked List node"""
        self.value = value
        self.prev = prev
        self.nxt = None
        if prev is not None:
            prev.nxt = self

    def mergewithnext(self):
        """Merges the value of this node with that of the current node, joining both nodes"""
        if self.nxt is None:
            raise ValueError("Node does not have a next node")
        self.value += self.nxt.value
        self.nxt = self.nxt.nxt
        if self.nxt is not None:
            self.nxt.prev = self

    def __str__(self):
        return str(self.value)
