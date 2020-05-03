(define (problem blocks-4-1)
(:domain blocks)
(:objects
	a b c d - block
	c1 c2 c3 c4 - column
)
(:init
	(hand-free)
	(clear b)
	(on b c)
	(on c a)
	(on a d)
	(bottom d c1)
	(empty c2)
	(empty c3)
	(empty c4)
)
(:goal (and
	(on d c)
	(on c a)
	(on a b)
))
)
