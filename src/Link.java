import java.util.Objects;

public class Link {
    private int from;
    private int to;

    public Link(int from, int to) {
        this.from = from;
        this.to = to;
    }

    public Link(Link o) {
        this.from = o.getFrom();
        this.to = o.getTo();
    }

    public int getFrom() {
        return from;
    }

    public int getTo() {
        return to;
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        } else if (!(o instanceof Link)) {
            return false;
        }
        Link link = (Link) o;
        return this.from == link.from &&
                this.to == link.to;
    }

    @Override
    public int hashCode() {
        return Objects.hash(from, to);
    }
}
