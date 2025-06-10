package linearmodel

import (
	"log"
	"math"
)

type LinearRegression struct {
	slope     float64
	intercept float64
	trained   bool
}

type Integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type Float interface {
	~float32 | ~float64
}

type Number interface {
	Integer | Float
}

func NewLinearRegression() LinearRegression {
	return LinearRegression{0, 0, false}
}

func Fit[T Number](lr *LinearRegression, x []T, y []T) {

	// Check if x and y are the same length
	if len(x) != len(y) {
		log.Println("x and y must be equal in length.")
		return
	}

	// n == number of samples. sum_x, sum_y, sum_xy, sum_xx used in final calculations of slope and intercept.
	n := T(len(x))
	sum_x := T(0)
	sum_y := T(0)
	sum_xy := T(0)
	sum_xx := T(0)

	for idx := range x {
		sum_x = sum_x + x[idx]
		sum_y = sum_y + y[idx]
		sum_xy = sum_xy + (x[idx] * y[idx])
		sum_xx = sum_xx + (x[idx] * x[idx])
	}

	lr.slope = float64((n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x))
	lr.intercept = float64((sum_y - T(lr.slope)*sum_x) / n)
	lr.trained = true

}

func Predict[T Number](lr *LinearRegression, x []T) []float64 {
	// Check if LinearRegression has been trained. If not, print warning and continue.
	if !lr.trained {
		log.Println("WARN: LinearRegression not trained!")
	}

	pred_y := make([]float64, len(x))

	for idx, value := range x {
		pred_y[idx] = float64(T(lr.slope)*value + T(lr.intercept))
	}

	return pred_y
}

func Score[T Number](lr *LinearRegression, x []T, y []T) float64 {
	// Calculates and returns the coefficient of determination.
	pred_y := Predict(lr, x)

	// Initialize variables for the residual sum of squares and total sum of squares.
	residual_squares := 0.0
	total_squares := 0.0

	y_mean := 0.0
	for idx := range y {
		y_mean = y_mean + float64(y[idx])
	}
	y_mean = y_mean / float64(len(y))

	for idx := range y {
		residual_squares = residual_squares + math.Pow(float64(y[idx])-pred_y[idx], 2)
		total_squares = total_squares + math.Pow(float64(y[idx])-y_mean, 2)
	}

	r_squared := 1 - (residual_squares / total_squares)
	return r_squared
}
