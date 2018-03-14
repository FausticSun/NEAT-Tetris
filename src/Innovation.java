public class Innovation extends Link implements Comparable<Innovation> {
    private int id;

    public Innovation(int id, Link link) {
        super(link);
        this.id = id;
    }

    public Innovation(Innovation o) {
        super(o);
        this.id = o.id;
    }

    public int getId() {
        return id;
    }

    @Override
    public int compareTo(Innovation o) {
        return Integer.compare(this.id, o.id);
    }
}
