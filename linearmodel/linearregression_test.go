package linearmodel

import "testing"

func TestFit(t *testing.T) {
	lr := NewLinearRegression()
	x := []int{0, 1, 2, 3, 4}
	y := []int{0, 1, 2, 3, 4}
	Fit(&lr, x, y)

	if lr.slope != 1 {
		t.Errorf("LinearRegression fit returned slope = %f; want 1", lr.slope)
	}

	if lr.intercept != 0 {
		t.Errorf("LinearRegression fit returned intercept = %f; want 0", lr.intercept)
	}
}

func TestPredict(t *testing.T) {
	lr := NewLinearRegression()
	x := []int{0, 1, 2, 3, 4}
	y := []int{0, 1, 2, 3, 4}
	Fit(&lr, x, y)

	pred_y := Predict(&lr, x)

	for idx := range y {
		if float64(y[idx]) != pred_y[idx] {
			t.Errorf("LinearRegression predict true_y and pred_y are not equal!")
		}
	}

}

func TestScore(t *testing.T) {
	lr := NewLinearRegression()
	x := []int{0, 1, 2, 3, 4}
	y := []int{0, 1, 2, 3, 4}
	Fit(&lr, x, y)

	r_squared := Score(&lr, x, y)

	if r_squared != 1 {
		t.Errorf("LinearRegression score returned %f; want 1", r_squared)
	}
}
