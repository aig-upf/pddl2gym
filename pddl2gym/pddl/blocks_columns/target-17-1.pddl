(define (problem target-17-0)
(:domain blocks)
(:objects
	a b - salient
	c d e f g h i j k l m n o p q - block
	c1 c2 c3 c4 c5 - column
)
(:init
	(holding q)
	(clear g)
	(clear h)
	(clear l)
	(clear p)
	(clear a)
	(on a j)
	(on j i)
	(on i b)
	(on b m)
	(on l f)
	(on f e)
	(on e k)
	(on g d)
	(on d c)
	(on c o)
	(on h n)
	(bottom m c1)
	(bottom k c2)
	(bottom o c3)
	(bottom n c4)
	(bottom p c5)
)
(:goal (and
	(on a b)
))
)
