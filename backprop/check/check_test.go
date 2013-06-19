package check

import (
	"testing"
)

func TestBackprop(t *testing.T) {
	for _, test := range Tests {
		bcopy := test.Src
		b := &bcopy
		b.Backward()
		if !eq(b.Delta, test.WantDelta) || !eq(b.DownErr, test.WantDownErr) {
			t.Errorf("%q: got\n\tDelta:   %v\n\tDownErr: %v\nwant\n\tDelta:   %v\n\tDownErr: %v", test.Name, b.Delta, b.DownErr, test.WantDelta, test.WantDownErr)
		}
	}
}
