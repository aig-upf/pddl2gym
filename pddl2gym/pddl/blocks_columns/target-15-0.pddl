(define (problem target-15-0)
(:domain blocks)
(:objects
	a b - salient
	c d e f g h i j k l m n o - block
	c1 c2 c3 c4 c5 - column
)
(:init
	(hand-free)
	(clear b)
	(clear e)
	(clear f)
	(clear i)
	(clear m)
	(on e j)
	(on j d)
	(on d l)
	(on l c)
	(on c g)
	(on m n)
	(on b a)
	(on a o)
	(on f k)
	(on i h)
	(bottom g c1)
	(bottom n c2)
	(bottom o c3)
	(bottom k c4)
	(bottom h c5)
)
(:goal (and
	(on a b)
))
)
