(define (domain blocks)

	(:requirements :typing)

	(:types
		salient - block
		block column - object
	)

	(:predicates
		(hand-free)
		(clear ?b - block)
		(holding ?b - block)
		(on ?b1 ?b2 - block)

		(bottom ?b - block ?col - column)
		(empty ?col - column)
	)

	(:action unstack 
		:parameters (?b1 ?b2 - block)
		:precondition (and
			(hand-free)
			(clear ?b1)
			(on ?b1 ?b2)
		) 
		:effect (and
			(not (hand-free))
			(not (clear ?b1))
			(clear ?b2)
			(holding ?b1)
			(not (on ?b1 ?b2))
		)
	)

	(:action stack
		:parameters (?b1 ?b2 - block)
		:precondition (and
			(clear ?b2)
			(holding ?b1)
		)
		:effect (and
			(hand-free)
			(clear ?b1)
			(not (clear ?b2))
			(not (holding ?b1))
			(on ?b1 ?b2)
		)
	)

	(:action pickup
		:parameters (?b - block ?col - column)
		:precondition (and
			(hand-free)
			(clear ?b)
			(bottom ?b ?col)
		)
		:effect (and
			(not (hand-free))
			(not (clear ?b))
			(holding ?b)
			(not (bottom ?b ?col))
			(empty ?col)
		)
	)

	(:action putdown
		:parameters (?b - block ?col - column)
		:precondition (and
			(holding ?b)
			(empty ?col)
		)
		:effect (and
			(hand-free)
			(clear ?b)
			(not (holding ?b))
			(bottom ?b ?col)
			(not (empty ?col))
		)
	)
)
