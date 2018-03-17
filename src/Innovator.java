import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Innovator {
    private Parameters params;
    private IdGenerator innovationIdGenerator = new IdGenerator();
    private IdGenerator neuronIdGenerator;
    private Map<Link, Innovation> linkInnovations = new HashMap<>();
    private Map<Link, List<Innovation>> nodeInnovations = new HashMap<>();

    public Innovator(Parameters params) {
        this.params = params;
        this.neuronIdGenerator = new IdGenerator(params.DEFAULT_NETWORK_SIZE);
    }

    public Gene innovateLink(Link link) {
        if (!linkInnovations.containsKey(link)) {
            linkInnovations.put(link, new Innovation(innovationIdGenerator.getNextId(), link));
        }
        return new Gene(params, linkInnovations.get(link));
    }

    public List<Gene> innovateNode(Link link) {
        if (!nodeInnovations.containsKey(link)) {
            List<Innovation> innovations = new ArrayList<>();
            int newNeuron = neuronIdGenerator.getNextId();
            innovations.add(new Innovation(innovationIdGenerator.getNextId(), new Link(link.getFrom(), newNeuron)));
            innovations.add(new Innovation(innovationIdGenerator.getNextId(), new Link(newNeuron, link.getTo())));
            nodeInnovations.put(link, innovations);
        }
        return nodeInnovations.get(link).stream().map(i -> new Gene(params, i)).collect(Collectors.toList());
    }

    public void clear() {
        linkInnovations.clear();
        nodeInnovations.clear();
    }
}
