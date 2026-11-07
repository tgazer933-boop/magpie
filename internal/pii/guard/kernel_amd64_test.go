package guard

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

// The model finds the same spans with the same scores, to the bit, with
// the AVX2 kernels and with SSE2 alone: a CPU with AVX2 masks a request
// as one without it does.
func TestModelSameWithoutAVX2(t *testing.T) {
	if !useAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	m, err := Load(dir(t))
	if err != nil {
		t.Fatal(err)
	}
	m.ml = nil
	var b strings.Builder
	for i := 0; b.Len() < 6<<10; i++ {
		fmt.Fprintf(&b, "/home/mrwl/projects/api/store_%d.go:%d: db.QueryRow(\"SELECT email FROM users WHERE id = $1\")\n", i, 40+i)
		fmt.Fprintf(&b, "Author: Wei Zhang <wei.zhang@example.com> postgres://app:S3cr3tPassw0rd%d@db.internal:5432/orders\n", i)
		fmt.Fprintf(&b, "联系人：张三，电话 138%08d，住在北京市海淀区中关村大街%d号。\n", i*7919, i)
	}
	text := b.String()
	with := m.Find(text)
	useAVX2 = false
	pickKernels()
	without := m.Find(text)
	useAVX2 = true
	pickKernels()
	if len(with) == 0 {
		t.Fatal("found nothing")
	}
	if !reflect.DeepEqual(with, without) {
		t.Fatalf("AVX2 found %d spans, SSE2 %d:\n%v\n%v", len(with), len(without), with, without)
	}
}

func TestDot4VNNIExact(t *testing.T) {
	if !useAVX512VNNI {
		t.Skip("no AVX512 VNNI on this CPU")
	}
	if dot4vnniFast == nil {
		t.Fatal("AVX512 VNNI kernel was not selected")
	}
	for _, k := range []int{64, 128, 384, 1536} {
		a := make([]uint8, k)
		b := make([]int8, 4*k)
		for i := range a {
			a[i] = uint8((i*73 + k) & 255)
			for c := range 4 {
				b[c*k+i] = int8((i*37+c*59)%256 - 128)
			}
		}
		qa := [...]uint8{0, 1, 128, 255}
		wb := [...]int8{-128, -1, 0, 127}
		for i := range qa {
			a[i] = qa[i]
			for c := range 4 {
				b[c*k+i] = wb[(i+c)%len(wb)]
			}
		}
		var got [4]int32
		dot4vnniFast(a, b, &got)
		for c := range 4 {
			var want int32
			for i, q := range a {
				want += int32(q) * int32(b[c*k+i])
			}
			if got[c] != want {
				t.Fatalf("k=%d column=%d: got %d want %d", k, c, got[c], want)
			}
		}
	}
}

func vnniTestText() string {
	text := "Contact zhangsan@example.com. "
	for len(text) < 4096 {
		text += "The ordinary service status is healthy and routine. "
	}
	return text[:4096]
}

func TestModelSameWithoutAVX512VNNI(t *testing.T) {
	if !useAVX512VNNI {
		t.Skip("no AVX512 VNNI on this CPU")
	}
	m, err := Load(dir(t))
	if err != nil {
		t.Fatal(err)
	}
	m.ml = nil
	text := vnniTestText()
	enabled := useAVX512VNNI
	t.Cleanup(func() {
		useAVX512VNNI = enabled
		pickKernels()
	})
	useAVX512VNNI = true
	pickKernels()
	with := m.Find(text)
	useAVX512VNNI = false
	pickKernels()
	without := m.Find(text)
	if len(with) == 0 {
		t.Fatal("VNNI model found nothing")
	}
	foundEmail := false
	for _, span := range with {
		if span.Label == "email" && text[span.Start:span.End] == "zhangsan@example.com" {
			foundEmail = true
		}
	}
	if !foundEmail {
		t.Fatalf("VNNI model did not find the email: %v", with)
	}
	if !reflect.DeepEqual(with, without) {
		t.Fatalf("VNNI found %d spans, legacy path %d:\n%v\n%v", len(with), len(without), with, without)
	}
}

func BenchmarkModelFindVNNI(b *testing.B) {
	if !useAVX512VNNI {
		b.Skip("no AVX512 VNNI on this CPU")
	}
	if !useAVX2 {
		b.Skip("no AVX2 baseline on this CPU")
	}
	modelDir := os.Getenv("MAGPIE_GUARD_DIR")
	if modelDir == "" {
		b.Skip("MAGPIE_GUARD_DIR is not set")
	}
	m, err := Load(modelDir)
	if err != nil {
		b.Fatal(err)
	}
	m.ml = nil
	text := vnniTestText()
	enabled := useAVX512VNNI
	defer func() {
		useAVX512VNNI = enabled
		pickKernels()
	}()
	for _, tc := range []struct {
		name string
		vnni bool
	}{{"VNNI", true}, {"AVX2", false}} {
		b.Run(tc.name, func(b *testing.B) {
			useAVX512VNNI = tc.vnni
			pickKernels()
			b.ReportAllocs()
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = m.Find(text)
			}
		})
	}
}
