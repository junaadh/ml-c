#set heading(numbering: "1.1")

#let avgsum(char1, char2) = $ 1/#char2 sum_(#char1=1)^#char2 $
#let pd(char) = $ \u{2202}_#char $

= GRADIENT DESCENT

$ C'(w) = lim_(e->0) (C(w + e) - C(w)) / e $

== Double

$
  C(w) &= avgsum(i, n)(x_i w - y_i)^2 \
  C'(w) 
    &= (avgsum(i, n) (x_i w -y_i)^2)' \ 
    &= 1/n (sum_(i=1)^n (x_i w -y_i)^2)' \ 
    &= 1/n ((x_0w - y_0)^2 + (x_1w - y_1)^2 + ... + (x_n w - y_n)^2)' \ 
    &= avgsum(i, n) ((x_i w -y_i)^2)' \ 
    &= avgsum(i, n) 2(x_i w -y_i)(x_i w -y_i)' \ 
    &= avgsum(i, n) 2(x_i w -y_i)(x_i w)' \ 
    &= avgsum(i, n) 2(x_i w -y_i)x_i ' \ 
$

$
  C(w) &= avgsum(i, n)(x_i w - y_i)^2 \
  C'(w) &= avgsum(i, n) 2(x_i w -y_i)x_i ' \ 
$

== One neuron model with 1 input

  #circle[
    $sigma$
  ]
