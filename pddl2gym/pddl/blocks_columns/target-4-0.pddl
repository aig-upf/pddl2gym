(define (problem target-4-0)
(:domain blocks)
(:objects
	a b - salient
	c d - block
	c1 c2 c3 c4 - column
)
(:init
	(hand-free)
	(clear a)
	(clear b)
	(clear c)
	(clear d)
	(bottom a c1)
	(bottom b c2)
	(bottom c c3)
	(bottom d c4)
)
(:goal (and
	(on a b)
))
)
