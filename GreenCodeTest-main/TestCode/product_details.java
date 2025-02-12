class Student3{
    int id;
    String name;

    String display(){ return id + " " + name; }

    public static void main(String args[]){
        Student3 s1 = new Student3();
        s1.id = 1;
        s1.name = "John";
        System.out.println(s1.display());

        Student3 s2 = new Student3();
        s2.id = 2;
        s2.name = "Jane";
        System.out.println(s2.display());
    }
}
