class A():
    def B(self,a:int) -> int:
        if a<2:
            print(a)
            return a
        return self.B(a-1)


my_a=A()
my_a.B(5)